import json
from flask.ext.cache import Cache
from plenario.settings import CACHE_CONFIG
from datetime import timedelta, date
from functools import update_wrapper
from flask import make_response, request, current_app
import csv
from shapely.geometry import asShape
from cStringIO import StringIO
from plenario.utils.helpers import get_size_in_degrees
from sqlalchemy.sql.schema import Table

import sqlalchemy as sa
from plenario.database import session, Base, app_engine as engine
from plenario.models import ShapeMetadata

cache = Cache(config=CACHE_CONFIG)

RESPONSE_LIMIT = 1000
CACHE_TIMEOUT = 60*60*6

class ParamValidator(object):

    def __init__(self):
        #self.whatever etc
        return

class ConditionBuilder(object):

    def __init__(self, dataset_name=None, shape_dataset_name=None):

        # general conditions include time, date, geometric location, etc.
        # as opposed to conditions on dataset-specific columns
        self.general_conditions = {}
        
        self.warnings = []
        
        # a mapping between query parameters and target values
        self.param_dict = {}

        if dataset_name:
            # Throws NoSuchTableError. Should be caught by caller.
            self.point_dataset = Table(dataset_name, Base.metadata, autoload=True,
                                 autoload_with=engine, extend_existing=True)
            self.point_cols = self.point_dataset.columns.keys()
            # SQLAlchemy boolean expressions
            self.conditions = []

            if shape_dataset_name is not None:
                self.set_shape(shape_dataset_name)

    def set_shape(self, shape_dataset_name):
        shape_table_meta = session.query(ShapeMetadata).get(shape_dataset_name)
        if shape_table_meta:
            shape_table = shape_table_meta.shape_table
            self.shape_cols = []
            self.shape_cols += ['{}.{}'.format(shape_table.name, key) for key in shape_table.columns.keys()]
            self.shape_dataset = shape_table

    def set_general_condition(self, name, general_cond, default):
        """
        :param name: Name of expected HTTP parameter
        :param transform: Function of type
                          f(param_val: str) -> (validated_argument: Option<object, None>, err_msg: Option<str, None>
                          Return value should be of form (output, None) if transformation was applied successully,
                          or (None, error_message_string) if transformation could not be applied.
        :param default: Value to apply to associate with parameter given by :name
                        if not specified by user. Can be a callable.
        :return: Returns the Validator to allow generative construction.
        """
        self.param_dict[name] = default
        self.general_conditions[name] = general_cond

        # For call chaining
        return self

    def make_conditions_from_params(self, params):
        for k, v in params.items():
            if k in self.general_conditions.keys():
                # k is a param name with a defined transformation
                # Get the transformation and apply it to v
                val, err = self.general_conditions[k](v)
                if err:
                    # v wasn't a valid string for this param name
                    return err
                # Override the default with the transformed value.
                self.param_dict[k] = val
                continue

            elif hasattr(self, 'cols'):

                '''
                TODO: will adding in condition tree parsing
                make this bit irrelevant?
                '''
                # Maybe k specifies a condition on the dataset
                cond, err = self._make_condition(k, v)
                print 'trying to make cond', k, v, cond, err
                # 'if cond' fails because sqlalchemy overrides __bool__
                if cond is not None:
                    self.conditions.append(cond)
                    continue
                elif err:
                    # Valid field was specified, but operator was malformed
                    return err
                # else k wasn't an attempt at setting a condition

            # This param is neither present in the optional params
            # nor does it specify a field in this dataset.
            if k != 'shape':
                #quick and dirty way to make sure 'shape' is not listed as an unused value
                warning = 'Unused parameter value "{}={}"'.format(k, v)
                self.warnings.append(warning)

        self._eval_defaults()

        self.make_nested_condition(params)
        return

    def get_geom(self):
        validated_geom = self.param_dict['location_geom__within']
        if validated_geom is not None:
            buff = self.param_dict.get('buffer', 100)
            return make_fragment_str(validated_geom, buff)

    def _eval_defaults(self):
        """
        Replace every value in vals that is callable with the returned value of that callable.
        Lets us lazily evaluate dafaults only when they aren't overridden.
        """
        for k, v in self.param_dict.items():
            if hasattr(v, '__call__'):
                self.param_dict[k] = v()


    def _make_condition(self, k, v, dset=None):
        # Generally, we expect the form k = [field]__[op]
        # Can also be just [field] in the case of simple equality
        tokens = k.split('__')
        # An attribute of the dataset
        field = tokens[0]
        '''
        if field not in self.cols and not self._check_shape_condition(field):
            # No column matches this key.
            # Rather than return an error here,
            # we'll return None to indicate that this field wasn't present
            # and let the calling function send a warning to the client.
            
            return None, None
        '''
        '''
        # the old code which, by default, searches the point dataset first, 
        # then the shape dataset, to find column
        col = self.point_dataset.columns.get(field)
        if col is None and self.param_dict.get('shape') is not None:
            col = self.param_dict['shape'].columns.get(field)
        '''

        # the new code which searches only the specified dataset 
        # or, by default, searches only the point dataset
        if dset != None:
            col = dset.columns.get(field)
            if col == None:
                #need to test this with fake column?
                return None, None

        else:
            col = self.point_dataset.columns.get(field)
            if col == None:
                #need to test this with fake column?
                return None, None

        if len(tokens) == 1:
            # One token? Then it's an equality operation of the form k=v
            # col == v creates a SQLAlchemy boolean expression
            return (col == v), None
        elif len(tokens) == 2:
            # Two tokens? Then it's of the form [field]__[op_code]=v
            op_code = tokens[1]
            valid_op_codes = self.field_ops.keys() + ['in']
            if op_code not in valid_op_codes:
                error_msg = "Invalid dataset field operator:" \
                                " {} called in {}={}".format(op_code, k, v)
                return None, error_msg
            else:
                cond = self._make_condition_with_operator(col, op_code, v)
                return cond, None

        else:
            error_msg = "Too many arguments on dataset field {}={}" \
                        "\n Expected [field]__[operator]=value".format(k, v)
            return None, error_msg

    def _make_condition_with_operator(self, col, op_code, target_value):
        if op_code == 'in':
            cond = col.in_(target_value.split(','))
            return cond
        else:   # Any other op code
            op_func = self.field_ops[op_code]
            # op_func is the name of a method bound to the SQLAlchemy column object.
            # Get the method and call it to create a binary condition (like name != 'Roy')
            # on the value the user specified.
            cond = getattr(col, op_func)(target_value)
            return cond

    def make_nested_condition(self, params):
        point_dataset_filters = params.get("point_dataset_filters")
        shape_dataset_filters = params.get("shape_dataset_filters")
        
        if point_dataset_filters:
            point_filter_dict = json.loads(point_dataset_filters)
            print point_filter_dict
            point_conditions = self.parse_condition_tree(point_filter_dict, self.point_dataset)
            self.conditions.append(point_conditions)

        if shape_dataset_filters:
            shape_filter_dict = json.loads(shape_dataset_filters)
            shape_conditions = self.parse_condition_tree(shape_filter_dict, self.shape_dataset)
            self.conditions.append(shape_conditions)

    def parse_condition_tree(self, json_condition_tree, dset):
        '''
        :param json_condition_tree: A dictionary representation of a tree of conditions
                                    of the following form - 
                                    {
                                        "op": [AND, OR. STMT]
                                        "val": ["left": json_condition_tree
                                                "right": json_condition_tree
                                               ],
                                               ["col_name": string
                                                "val": value
                                               ]
                                    }
        :return: condition object that is useable by SQLAlchemy
        '''

        op = json_condition_tree.get("op")
        if op == "STMT":
            k = json_condition_tree["val"]["col_name"]# column name
            v = json_condition_tree["val"]["val"]# value
            self._make_condition(k, v, dset)
            #make condition
            pass
        if op == "AND":
            left = self.parse_condition_tree(json_condition_tree["val"]["left"], dset)
            right = self.parse_condition_tree(json_condition_tree["val"]["right"], dset)
            #and together
        if op == "OR":
            left = self.parse_condition_tree(json_condition_tree["val"]["left"], dset)
            right = self.parse_condition_tree(json_condition_tree["val"]["right"], dset)
            #or together
        else:
            #error
            pass


    def form_query_from_conditions(self, aggregate_points=False):
        dset = self.point_dataset
        try:
            q = session.query(dset)
            if self.conditions:
                #TODO: split this into point and shape conditions?
                #print 'validator conditions', validator.conditions, len(validator.conditions)
                q = q.filter(*self.conditions)
        except Exception as e:
            return internal_error('Failed to construct column filters.', e)

        try:
            # Add time filters
            maker = FilterMaker(self.param_dict, dataset=dset)
            q = q.filter(*maker.time_filters())

            # Add geom filter, if provided
            geom = self.get_geom()
            if geom is not None:
                geom_filter = maker.geom_filter(geom)
                q = q.filter(geom_filter)
        except Exception as e:
            return internal_error('Failed to construct time and geometry filters.', e)

        #if the query specified a shape dataset, add a join to the sql query with that dataset
        shape_table = self.shape_dataset #self.param_dict.get('shape')
        if shape_table != None:
            shape_columns = ['{}.{} as {}'.format(shape_table.name, col.name, col.name) for col in shape_table.c]   
            if aggregate_points: 
                q = q.from_self(shape_table).filter(dset.c.geom.ST_Intersects(shape_table.c.geom)).group_by(shape_table)
            else:
                q = q.join(shape_table, dset.c.geom.ST_Within(shape_table.c.geom))
                #add columns from shape dataset to the select statement
                q = q.add_columns(*shape_columns)

        #print q
        return q

def unknownObjectHandler(obj):
    #convert Plenario objects into json for response; currently handles geoms and dates
    if type(obj) == Table:
        return obj.name
    elif isinstance(obj, date):
        return obj.isoformat()
    else:
        raise ValueError

def dthandler(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    else:
        raise ValueError


# http://flask.pocoo.org/snippets/56/
def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True): # pragma: no cover
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


def make_cache_key(*args, **kwargs):
    path = request.path
    args = str(hash(frozenset(request.args.items())))
    return (path + args).encode('utf-8')


def make_csv(data):
    outp = StringIO()
    writer = csv.writer(outp)
    writer.writerows(data)
    return outp.getvalue()


def extract_first_geometry_fragment(geojson):
    """
    Given a geojson document, return a geojson geometry fragment marked as 4326 encoding.
    If there are multiple features in the document, just make a fragment of the first feature.
    This is what PostGIS's ST_GeomFromGeoJSON expects.
    :param geojson: A full geojson document
    :type geojson: str
    :return: dict representing geojson structure
    """
    geo = json.loads(geojson)
    if 'features' in geo.keys():
        fragment = geo['features'][0]['geometry']
    elif 'geometry' in geo.keys():
        fragment = geo['geometry']
    else:
        fragment = geo

    return fragment


def make_fragment_str(geojson_fragment, buffer=100):
    if geojson_fragment['type'] == 'LineString':
        shape = asShape(geojson_fragment)
        lat = shape.centroid.y
        x, y = get_size_in_degrees(buffer, lat)
        geojson_fragment = shape.buffer(y).__geo_interface__

    geojson_fragment['crs'] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    return json.dumps(geojson_fragment)

class FilterMaker(object):
    """
    Given dictionary of validated arguments and a sqlalchemy table,
    generate binary consitions on that table restricting time and geography.
    Can also create a postgres-formatted geography for further filtering
    with just a dict of arguments.
    """

    def __init__(self, args, dataset=None):
        """
        :param args: dict mapping arguments to values as taken from a Validator
        :param dataset: table object of particular dataset being queried, if available
        """
        self.args = args
        self.dataset = dataset

    def time_filters(self):
        """
        :return: SQLAlchemy conditions derived from time arguments on :dataset:
        """
        filters = []
        d = self.dataset
        try:
            lower_bound = d.c.point_date >= self.args['obs_date__ge']
            filters.append(lower_bound)
        except KeyError:
            pass

        try:
            upper_bound = d.c.point_date <= self.args['obs_date__le']
            filters.append(upper_bound)
        except KeyError:
            pass

        try:
            start_hour = self.args['date__time_of_day_ge']
            if start_hour != 0:
                lower_bound = sa.func.date_part('hour', d.c.point_date).__ge__(start_hour)
                filters.append(lower_bound)
        except KeyError:
            pass

        try:
            end_hour = self.args['date__time_of_day_le']
            if end_hour != 23:
                upper_bound = sa.func.date_part('hour', d.c.point_date).__ge__(end_hour)
                filters.append(upper_bound)
        except KeyError:
            pass

        return filters

    def geom_filter(self, geom_str):
        """
        :param geom_str: geoJSON string from Validator ready to throw into postgres
        :return: geographic filter based on location_geom__within and buffer parameters
        """
        # Demeter weeps
        return self.dataset.c.geom.ST_Within(sa.func.ST_GeomFromGeoJSON(geom_str))


