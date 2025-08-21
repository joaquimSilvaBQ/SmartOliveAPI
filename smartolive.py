from flask import Flask, request, jsonify
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import cv2
import numpy as np
import time
import boto3
import os
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import smtplib
import io
from PIL import Image
import requests
from datetime import datetime, timedelta
import locale
from models import db, Companies, IrrigationHouses, Cameras, Tanks, TankStats

__author__ = 'Joaquim Silva'

# Configure locale
locale.setlocale(locale.LC_TIME, 'pt_PT.utf8')

# Flask app configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DB_CONNECTION_STRING')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the SQLAlchemy db instance
db.init_app(app)


class ImageProcessor:
    """Handles image processing operations for computer vision tasks."""
    
    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
        """Apply brightness and contrast adjustments to an image."""
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    @staticmethod
    def draw_text(img, text, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN,
                  font_scale=4, font_thickness=4, text_color=(0, 255, 0),
                  text_color_bg=(0, 0, 0)):
        """Draw text on an image with background."""
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), 
                   font, font_scale, text_color, font_thickness)
        return text_size

    @staticmethod
    def resize_image(image, width=1600, height=1300):
        """Resize image to specified dimensions."""
        down_points = (width, height)
        return cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def convert_to_bytes(image):
        """Convert OpenCV image to bytes for storage/transmission."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb, mode='RGB')
        img_byte_array = io.BytesIO()
        pil_image.save(img_byte_array, format='JPEG')
        img_byte_array.seek(0)
        return img_byte_array


class GeometryUtils:
    """Utility functions for geometric operations."""
    
    @staticmethod
    def is_inside_roi(point, top_left, bottom_right):
        """Check if a point is inside the bounding box."""
        x, y = point[0]
        return (top_left[0] <= x <= bottom_right[0] and 
                top_left[1] <= y <= bottom_right[1])
    
    @staticmethod
    def is_contour_inside_roi(contour, roi_vertices):
        """Check if contour is inside the ROI using shapely."""
        polygon = Polygon(roi_vertices)
        contour_point = Point(contour[0])
        return polygon.contains(contour_point)


class TankAnalyzer:
    """Analyzes tank levels using computer vision."""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.main_tank_capacity = 9030
        self.top_tank_capacity = 970
    
    def detect_red_balls(self, enhanced_img_hsv, tank_data):
        """Detect red balls in the image using HSV color filtering."""
        min_red_balls = np.array(tank_data['min_red_ball_mask'])
        max_red_balls = np.array(tank_data['max_red_ball_mask'])
        mask_red_balls = cv2.inRange(enhanced_img_hsv, min_red_balls, max_red_balls)

        # Apply dilation to the contour mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_redball_mask = cv2.dilate(mask_red_balls, kernel, iterations=1)
        
        return cv2.findContours(dilated_redball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def calculate_tank_level(self, contour, roi_coordinate, frame):
        """Calculate tank level based on red ball position."""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw box around red ball
        topleft, bottomright = (x - 10, y - 10), (x + w + 10, y + h + 10)
        cv2.rectangle(frame, topleft, bottomright, (0, 0, 255), 4)

        # Draw line on red ball center
        center_x = x + w // 2
        center_y = y + h // 2
        line_start = (x + 70, center_y)
        line_end = (x + w, center_y)
        cv2.line(frame, line_start, line_end, (0, 0, 255), 1)

        # Calculate percentage filled
        roi_bottom = roi_coordinate[3][1]
        roi_height = roi_coordinate[3][1] - roi_coordinate[0][1]
        percentage_filled = round(((roi_bottom - center_y) / roi_height) * 100, 2)

        # Calculate liters based on tank type
        if roi_height > 400:
            liters = int((percentage_filled / 100) * self.main_tank_capacity)
        else:
            liters = int((percentage_filled / 100) * self.top_tank_capacity) + self.main_tank_capacity

        return f"{liters} litros", liters

    def process_tank(self, frame, tank_data, enhanced_img_hsv):
        """Process a single tank to detect red balls and calculate level."""
        red_balls_contours, _ = self.detect_red_balls(enhanced_img_hsv, tank_data)
        
        # Define ROIs
        main_tank_coordinates = np.array([list(item) for item in tank_data['main_tank_coordinates']])
        doomed_tank_coordinates = np.array([list(item) for item in tank_data['doomed_tank_coordinates']])
        
        # Draw ROI boundaries
        cv2.polylines(frame, [main_tank_coordinates], True, (0, 128, 0), 4)
        cv2.polylines(frame, [doomed_tank_coordinates], True, (0, 128, 0), 4)

        rois_coordinates = [main_tank_coordinates, doomed_tank_coordinates]
        
        for contour in red_balls_contours:
            for roi_coordinate in rois_coordinates:
                if GeometryUtils.is_contour_inside_roi(contour, roi_coordinate):
                    liters_string, liters_value = self.calculate_tank_level(contour, roi_coordinate, frame)
                    ImageProcessor.draw_text(frame, liters_string, 
                                           (roi_coordinate[0][0] - 200, roi_coordinate[0][1] - 200))
                    return liters_string, liters_value
        
        return None, None


class IrrigationMetricsCalculator:
    """Calculates irrigation metrics and statistics."""
    
    @staticmethod
    def calculate_irrigation_metrics(tank, calculated_liters):
        """Calculate irrigation metrics for a tank."""
        irrigation_happened = False
        
        # Get last reading where irrigation happened
        last_reading_with_irrigation = TankStats.query.filter_by(
            irrigation_happened=True, tank_id=tank.id
        ).order_by(TankStats.date.desc()).first()
        
        if last_reading_with_irrigation:
            # Check if current reading represents meaningful change
            if calculated_liters < last_reading_with_irrigation.liters - 25:
                tank.stored_liquid_liters = calculated_liters
                irrigation_happened = True
        else:
            tank.stored_liquid_liters = calculated_liters
            irrigation_happened = True

        # Create new tank stat
        new_tank_stat = TankStats(
            tank_id=tank.id, 
            date=datetime.now(), 
            liters=calculated_liters, 
            irrigation_happened=irrigation_happened
        )
        db.session.add(new_tank_stat)
        db.session.commit()

        return irrigation_happened

    @staticmethod
    def get_weekly_irrigation_data(tank_id):
        """Get irrigation data for the last week."""
        today = datetime.today()
        last_week = today - timedelta(days=7)
        
        current_week_irrigations = TankStats.query.filter(
            TankStats.date.between(last_week, today),
            TankStats.irrigation_happened == True,
            TankStats.tank_id == tank_id
        ).order_by(TankStats.date.asc()).all()

        return current_week_irrigations

    @staticmethod
    def calculate_weekly_averages(current_week_irrigations):
        """Calculate weekly irrigation averages."""
        if len(current_week_irrigations) <= 1:
            return None, []

        # Calculate average liters per irrigation
        irrigation_liters_sum = 0
        for i in range(1, len(current_week_irrigations)):
            irrigation_liters_sum += current_week_irrigations[i - 1].liters - current_week_irrigations[i].liters

        irrigation_liters_week_average = int(irrigation_liters_sum / (len(current_week_irrigations) - 1))
        
        # Get days when irrigation happened
        dates_when_irrigation_happened = [record.date for record in current_week_irrigations]
        days_where_irrigation_happened = [
            date.strftime('%A').lower() for date in dates_when_irrigation_happened
        ]

        return irrigation_liters_week_average, days_where_irrigation_happened


class EmailService:
    """Handles email sending functionality."""
    
    def __init__(self):
        self.email_sender = 'info@smartolive.solutions'
        self.email_password = os.getenv('email_password')
        self.email_receivers = ['joaquim+1@smartolive.solutions']
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587

    def create_tank_strings(self, irrigation_metrics):
        """Create HTML strings for each tank's data."""
        tank_strings = []
        for tank_id, info in irrigation_metrics['tanks_data'].items():
            irrigation_liters_week_average = info.get('irrigation_liters_week_average')
            days_where_irrigation_happened = info.get('days_where_irrigation_happened')
            
            if irrigation_liters_week_average is not None:
                tank_strings.append(
                    f"Depósito {tank_id}: <b>{info['current_liters']} litros</b>, "
                    f"<b>{irrigation_liters_week_average} litros</b> de média por rega esta semana."
                )
            else:
                tank_strings.append(
                    f"Depósito {tank_id}: <b>{info['current_liters']} litros</b>. "
                    f"Média de litros por rega apenas disponivel para a próxima rega."
                )

            if days_where_irrigation_happened:
                irrigation_days_string = ', '.join(
                    f'<strong>{day}</strong>' for day in days_where_irrigation_happened
                )
                tank_strings.append(f"Este depósito regou nos dias: {irrigation_days_string}.")

        return tank_strings

    def send_reading_email(self, irrigation_metrics, img_byte_array):
        """Send email with irrigation reading data."""
        tank_strings = self.create_tank_strings(irrigation_metrics)
        
        subject = 'Leitura automática depósitos Ferreira'
        body = f"""\
            <html>
            <body>
                {"".join(f"<p>{tank_string}</p>" for tank_string in tank_strings)}
                <p>Prefazendo um total de <b>{irrigation_metrics['total_liters']} litros em stock.</b></p>
                <img src="cid:deposito.png">
            </body>
            </html>
        """

        for email_receiver in self.email_receivers:
            self._send_single_email(email_receiver, subject, body, img_byte_array)

        print('Emails sent successfully')

    def _send_single_email(self, email_receiver, subject, body, img_byte_array):
        """Send a single email."""
        email = MIMEMultipart()
        email['From'] = self.email_sender
        email['To'] = email_receiver
        email['Subject'] = subject

        # Email body
        email.attach(MIMEText(body, 'html'))
        
        # Attach image
        img_byte_array.seek(0)
        img = MIMEImage(img_byte_array.read())
        img.add_header('Content-ID', '<deposito.png>')
        email.attach(img)

        # Send the email
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(self.email_sender, self.email_password)
        server.sendmail(self.email_sender, email_receiver, email.as_string())
        server.quit()


class StorageService:
    """Handles file storage operations."""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ['aws_access_key_id'],
            aws_secret_access_key=os.environ['aws_secret_access_key'],
            region_name='eu-west-3'
        )
        self.bucket_name = 'depositos-fertilizante'

    def store_reading(self, img_byte_array):
        """Store image reading to S3."""
        image_name = f'deposito_{time.time()}_docker.png'

        # Generate the presigned URL
        response = self.s3_client.generate_presigned_post(
            Bucket=self.bucket_name,
            Key=image_name,
            ExpiresIn=35
        )

        # Upload file to S3 using presigned URL
        img_byte_array.seek(0)
        files = {'file': img_byte_array}
        
        try:
            requests.post(response['url'], data=response['fields'], files=files)
            print("Image saved successfully")
        except Exception as e:
            print(f"Error saving image: {e}")


class CameraProcessor:
    """Processes camera feeds and analyzes tank levels."""
    
    def __init__(self):
        self.tank_analyzer = TankAnalyzer()
        self.metrics_calculator = IrrigationMetricsCalculator()
        self.email_service = EmailService()
        self.storage_service = StorageService()

    def process_camera(self, camera):
        """Process a single camera and its associated tanks."""
        camera_data = camera.to_dict()
        time.sleep(1)
        
        filled_liters = []
        img_byte_array = io.BytesIO()
        
        print(f'Processing camera with {len(camera.tanks)} tanks')
        
        # make sure all tanks are tracked
        while len(filled_liters) != len(camera.tanks):
            try:
                video = cv2.VideoCapture(
                    f"rtsp://{os.getenv('rstp_credentials')}@{camera_data['rstp_stream_ip']}/stream1"
                )
                success, frame = video.read()
                
                if success:
                    # Enhance image for better ball detection
                    image_with_more_contrast = ImageProcessor.apply_brightness_contrast(frame, 0, 64)
                    enhanced_img_hsv = cv2.cvtColor(image_with_more_contrast, cv2.COLOR_BGR2HSV)

                    filled_liters = self._process_tanks(frame, camera.tanks, enhanced_img_hsv)
                
                video.release()
                
            except Exception as e:
                print(f"Error processing camera: {e}")
                break

        return filled_liters, frame, img_byte_array

    def _process_tanks(self, frame, tanks, enhanced_img_hsv):
        """Process all tanks for a camera."""
        filled_liters = []
        
        for tank in tanks:
            tank_data = tank.to_dict()
            liters_string, liters_value = self.tank_analyzer.process_tank(
                frame, tank_data, enhanced_img_hsv
            )
            
            if liters_string:
                filled_liters.append(liters_string)
        
        return filled_liters

    def calculate_irrigation_metrics(self, camera, filled_liters):
        """Calculate irrigation metrics for all tanks in a camera."""
        irrigation_metrics = {
            'total_liters': 0,
            'tanks_data': {}
        }

        for index, liters in enumerate(filled_liters):
            calculated_liters = int(liters.split()[0])
            current_tank = camera.tanks[index]
            
            # Store current tank data
            irrigation_metrics['tanks_data'][current_tank.id] = {
                'current_liters': calculated_liters
            }
            irrigation_metrics['total_liters'] += calculated_liters
            
            # Calculate irrigation metrics
            self.metrics_calculator.calculate_irrigation_metrics(current_tank, calculated_liters)
            
            # Get weekly data
            current_week_irrigations = self.metrics_calculator.get_weekly_irrigation_data(current_tank.id)
            irrigation_liters_week_average, days_where_irrigation_happened = (
                self.metrics_calculator.calculate_weekly_averages(current_week_irrigations)
            )
            
            irrigation_metrics['tanks_data'][current_tank.id]['days_where_irrigation_happened'] = (
                days_where_irrigation_happened
            )
            
            if irrigation_liters_week_average is not None:
                irrigation_metrics['tanks_data'][current_tank.id]['irrigation_liters_week_average'] = (
                    irrigation_liters_week_average
                )

        return irrigation_metrics


# Flask route handlers
@app.before_request
def require_api_key_and_params():
    """Middleware to check for required access parameter."""
    required_param = request.args.get('parametro_acesso')
    if not required_param or required_param != 'ferreira':
        response = jsonify({"message": "Argumento de acesso em falta"})
        response.status_code = 400
        return response


@app.route('/')
def home():
    """Home route to display basic information."""
    companies = Companies.query.all()
    print('companies ', companies)
    cameras = Cameras.query.all()
    print('cameras ', cameras[0].to_dict()['rstp_stream_ip'])
    for camera in cameras:
        print(camera.tanks[0].to_dict())

    return "Olá."


@app.route('/depositos-ferreira')
def analyze_depositos_ferreira():
    """Main route to analyze tank levels and send email report."""
    try:
        cameras = Cameras.query.all()
        camera_processor = CameraProcessor()
        
        for camera in cameras:
            filled_liters, frame, img_byte_array = camera_processor.process_camera(camera)
            
            if len(filled_liters) == len(camera.tanks):
                # Calculate metrics
                irrigation_metrics = camera_processor.calculate_irrigation_metrics(camera, filled_liters)
                
                # Process image for storage/email
                resized_image = ImageProcessor.resize_image(frame)
                img_byte_array = ImageProcessor.convert_to_bytes(resized_image)
                
                # Store image (commented out)
                # camera_processor.storage_service.store_reading(img_byte_array)
                
                # Send email
                camera_processor.email_service.send_reading_email(irrigation_metrics, img_byte_array)
                
                # Return success response
                percentages_text = ", ".join(map(str, filled_liters))
                return f"""\
                    <html>
                    <body>
                        <h2>Email com leituras enviado com sucesso!</h2>
                        <h3>{'Os depósitos estão com os seguintes niveis: ' if len(filled_liters) > 1 else 'O Depósito está com o nivel de '}{percentages_text}</h3>
                        <h3>Página em construção... Obrigado pela sua colaboração!</h3>
                    </body>
                    </html>
                """
        
        return "No cameras found or processing failed."
        
    except Exception as e:
        print(f"Error in depositos-ferreira route: {e}")
        return "Error processing request", 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
