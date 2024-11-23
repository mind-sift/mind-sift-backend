from app.server import app
from mangum import Mangum

handler = Mangum(app)