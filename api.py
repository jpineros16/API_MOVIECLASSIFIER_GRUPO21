#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from prediction import predict_genres

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Movie Genre Prediction API',
    description='Multi-label movie genre classification API (24 genres)')

ns = api.namespace('predict',
     description='Movie Genre Predictor')

parser = api.parser()
parser.add_argument('title', type=str, required=True,
                    help='Movie title',                  location='args')
parser.add_argument('plot',  type=str, required=True,
                    help='Movie plot / synopsis',        location='args')
parser.add_argument('year',  type=int, required=True,
                    help='Release year (e.g. 1995)',     location='args')

@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    def get(self):
        args = parser.parse_args()
        return predict_genres(args), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
