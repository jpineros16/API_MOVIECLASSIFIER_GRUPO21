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

# Respuesta: probabilidad [0-1] para cada uno de los 24 géneros
resource_fields = api.model('Resource', {
    'Action':       fields.Float,
    'Adventure':    fields.Float,
    'Animation':    fields.Float,
    'Biography':    fields.Float,
    'Comedy':       fields.Float,
    'Crime':        fields.Float,
    'Documentary':  fields.Float,
    'Drama':        fields.Float,
    'Family':       fields.Float,
    'Fantasy':      fields.Float,
    'Film-Noir':    fields.Float,
    'History':      fields.Float,
    'Horror':       fields.Float,
    'Music':        fields.Float,
    'Musical':      fields.Float,
    'Mystery':      fields.Float,
    'News':         fields.Float,
    'Romance':      fields.Float,
    'Sci-Fi':       fields.Float,
    'Short':        fields.Float,
    'Sport':        fields.Float,
    'Thriller':     fields.Float,
    'War':          fields.Float,
    'Western':      fields.Float,
})

@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        return predict_genres(args), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
