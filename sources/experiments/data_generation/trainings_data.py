
import json
import numpy as np
import datetime

from sources.experiments.ellipsis_data_support.make_ellipsis import create_ellipsis_grid
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution


def make_permeability_matrix_one(gridWidth, gridHeight, innerGridWidth, innerGridHeight, majorSemiAxis, minorSemiAxis, permeability, angle):
    eps_data = create_ellipsis_grid(gridWidth, gridHeight, innerGridWidth, innerGridHeight, majorSemiAxis,\
                                    minorSemiAxis, permeability, angle)
    return eps_data

def encode_ndarray(array, columns, rows):
    return array.tolist()
    #resulting_array = []
    #current_array = []
    #i = 0
    #for value in np.nditer(array):
    #    if i==0:
    #        current_array = []
    #        resulting_array.append(current_array)
    #   current_array.append(value.item(0))
    #    i += 1
    #    if i == columns:
    #        i = 0
    #return resulting_array

def as_ndarray(array):
    return np.asarray(array, dtype=float)


def encode_TrainingsSet(obj):
    if isinstance(obj, TrainingsSet):
        return ''
    else:
        type_name = obj.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

def as_TrainingsSet(json_data):
    if '__TrainingsSet__' in json_data:
        return TrainingsSetDecoder().decode(json_data)
    return json_data


class TrainingsSetEncoder(json.JSONEncoder):
    def default(self, data):
        if isinstance(data, TrainingsSet):
            return data.encode()
        elif isinstance(data, TrainingsSetGeometry):
            return data.encode()
        elif isinstance(data, ChargeDistribution):
            return data.chargesList
        else:
            super().default(self, data)

class TrainingsSetDecoder:
    def decode(self, json_data):

        count = json_data["count"]
        items = json_data["items"]
        label = json_data["label"]
        geometry = json_data["geometry"]
        chargesList = json_data["charges"]
        timestamp = json_data["createdAt"]
        trainingsSet = self.init_data(TrainingsSetGeometry(geometry), chargesList, count, label, timestamp)
        trainingsSet.decode(items)
        return trainingsSet

    def init_data(self, geometry, chargesList, count, label, timestamp):
        semiMajorAxis = [None] * count
        semiMinorAxis = [None] * count
        permittivities = [None] * count
        angles = [None] * count
        return TrainingsSet(geometry, chargesList, semiMajorAxis, semiMinorAxis, permittivities, angles, label=label, timestamp=timestamp)

class TrainingsSetGeometry:
    def __init__(self, *args, **kwargs):
        self.gridWidth = (kwargs['gridWidth'] if 'gridWidth' in kwargs else args[0][0])
        self.gridHeight = (kwargs['gridHeight'] if 'gridHeight' in kwargs else args[0][1])
        self.innerGridWidth = (kwargs['innerGridWidth'] if 'innerGridWidth' in kwargs else args[0][2])
        self.innerGridHeight = (kwargs['innerGridHeight'] if 'innerGridHeight' in kwargs else args[0][3])

    def encode(self):
        return [self.gridWidth, self.gridHeight, self.innerGridWidth, self.innerGridHeight ]

class TrainingsElement:
    def __init__(self, trainingsSet, index):
        self.trainingsSet = trainingsSet
        self.index = index

    def get_semiMajorAxis(self):
        return self.trainingsSet.semiMajorAxises[self.index]

    def set_semiMajorAxis(self, value):
        self.trainingsSet.semiMajorAxises[self.index] = value

    def get_semiMinorAxis(self):
        return self.trainingsSet.semiMinorAxises[self.index]

    def set_semiMinorAxis(self, value):
        self.trainingsSet.semiMinorAxises[self.index] = value

    def get_permittivity(self):
        return self.trainingsSet.permittivities[self.index]

    def set_permittivity(self, value):
        self.trainingsSet.permittivities[self.index] = value

    def get_angle(self):
        return self.trainingsSet.angles[self.index]

    def set_angle(self, value):
        self.trainingsSet.angles[self.index] = value

    def calc_permittivity_matrix(self, gridWidth, gridHeight, innerGridWidth, innerGridHeight):
        self.trainingsSet.permittivity_matrix[self.index] = make_permeability_matrix_one(gridWidth, gridHeight, innerGridWidth, innerGridHeight,
                                                                majorSemiAxis=self.get_semiMajorAxis(),
                                                                minorSemiAxis=self.get_semiMinorAxis(),
                                                                permeability=self.get_permittivity(),
                                                                angle=self.get_angle())

    def is_permittivity_matrix_calculated(self):
        return self.trainingsSet.permittivity_matrix[self.index] != None

    def get_permittivity_matrix(self):
        return self.trainingsSet.permittivity_matrix[self.index]


class TrainingsSet:
    def __init__(self, geometry, chargesList, semiMajorAxises, semiMinorAxises, permittivities, angles, label=None, timestamp=None):
        self.geometry = geometry
        self.chargesList = chargesList
        self.semiMajorAxises = semiMajorAxises
        self.semiMinorAxises = semiMinorAxises
        self.permittivities = permittivities
        self.angles = angles
        c = len(permittivities)
        self.permittivity_matrix = [None] * len(permittivities)
        self.label = label
        self.timestamp = timestamp or datetime.datetime.utcnow()

    def count(self):
        return len(self.permittivities)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.angles):
            element = TrainingsElement(self, self.index)
            self.index += 1
            return element
        else:
            raise StopIteration

    def encode(self):
        items = []
        for item in self:
            items.append({'index':item.index+1,
                          'semiMajorAxis':item.get_semiMajorAxis(),
                          'semiMinorAxis':item.get_semiMinorAxis(),
                          'eps':item.get_permittivity(),
                          'permittivity_matrix':encode_ndarray(item.get_permittivity_matrix(), int(self.geometry.gridWidth), int(self.geometry.gridHeight)),
                          'angle':item.get_angle()
                          })

        return { '__TrainingsSet__':True, 'label':self.label, 'createdAt':str(self.timestamp), 'geometry':self.geometry, 'charges':self.chargesList, 'count': self.count(), 'items':items }

    def decode(self, itemsArray):
        for item in self:
            itemInArray = itemsArray[item.index]
            item.set_semiMajorAxis(itemInArray["semiMajorAxis"])
            item.set_semiMinorAxis(itemInArray["semiMinorAxis"])
            item.set_permittivity(itemInArray["eps"])
            item.set_angle(itemInArray["angle"])
            self.permittivity_matrix[item.index] = as_ndarray(itemInArray["permittivity_matrix"])