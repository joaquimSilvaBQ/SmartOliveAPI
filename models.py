from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import ARRAY

db = SQLAlchemy()

class Companies(db.Model):
    __tablename__ = 'companies'

    id = db.Column(db.Integer, primary_key=True)
    email_address = db.Column(db.String(255), unique=True, nullable=False)
    phone_number = db.Column(db.String(20))
    name = db.Column(db.String(100))

class IrrigationHouses(db.Model):
    __tablename__ = 'irrigationhouses'

    id = db.Column(db.Integer, primary_key=True)
    irrigationhouse_id = db.Column(db.Integer)
    rstp_stream_ip = db.Column(db.String(255))

class Cameras(db.Model):
    __tablename__ = 'cameras'

    id = db.Column(db.Integer, primary_key=True)
    irrigationhouse_id = db.Column(db.Integer, db.ForeignKey('irrigationhouses.id'), nullable=False)
    rstp_stream_ip = db.Column(db.String(255))

    tanks = db.relationship('Tanks', back_populates='camera', cascade="all, delete-orphan", order_by='Tanks.id')

    def to_dict(self):
        return {
            'id': self.id,
            'irrigationhouse_id': self.irrigationhouse_id,
            'rstp_stream_ip': self.rstp_stream_ip,
            'tanks': [tank.to_dict() for tank in self.tanks]
        }

class Tanks(db.Model):
    __tablename__ = 'tanks'

    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=False)
    left_or_right_order = db.Column(db.Integer)
    stored_liquid_liters = db.Column(db.Numeric)
    min_red_ball_mask = db.Column(ARRAY(db.Integer))
    max_red_ball_mask = db.Column(ARRAY(db.Integer))
    main_tank_coordinates = db.Column(ARRAY(db.Integer))
    doomed_tank_coordinates = db.Column(ARRAY(db.Integer))

    camera = db.relationship('Cameras', back_populates='tanks')
    tankstats = db.relationship('TankStats', back_populates='tank')

    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'left_or_right_order': self.left_or_right_order,
            'stored_liquid_liters': self.stored_liquid_liters,
            'min_red_ball_mask': self.min_red_ball_mask,
            'max_red_ball_mask': self.max_red_ball_mask,
            'main_tank_coordinates': self.main_tank_coordinates,
            'doomed_tank_coordinates': self.doomed_tank_coordinates,
        }
    
class TankStats(db.Model):
    __tablename__ = 'tankstats'

    id = db.Column(db.Integer, primary_key=True)
    tank_id = db.Column(db.Integer, db.ForeignKey('tanks.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    liters = db.Column(db.Numeric, nullable=False)
    irrigation_happened = db.Column(db.Boolean, default=False, nullable=False)

    tank = db.relationship('Tanks', back_populates='tankstats')

    def to_dict(self):
        return {
            'id': self.id,
            'tank_id': self.tank_id,
            'date': self.date,
            'liters': self.liters,
        }