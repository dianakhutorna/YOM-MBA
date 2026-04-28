from mangum import Mangum

from training.src.scripts.serve_recommendations_api import app


handler = Mangum(app)