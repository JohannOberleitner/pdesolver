
import numpy as np
import datetime
import json

def encode_ndarray(array):
    return array.tolist()

def as_ndarray(array):
    return np.asarray(array, dtype=float)

def as_ResultsSet(json_data):
    if '__ResultsSet__' in json_data:
        return ResultsSetDecoder().decode(json_data)
    return json_data

class ResultsSetEncoder(json.JSONEncoder):
    def default(self, data):
        if isinstance(data, ResultsSet):
            return data.encode()
        else:
            super().default(self, data)

class ResultsSetDecoder:
    def decode(self, json_data):

        count = json_data["count"]
        items = json_data["items"]
        label = json_data["label"]
        timestamp = json_data["createdAt"]
        resultsSet = ResultsSet(label, timestamp)
        resultsSet.decode(items)
        return resultsSet

class ResultsElement:
    def __init__(self, resultsSet, index):
        self.resultsSet = resultsSet
        self.index = index

    def get_values(self):
        return self.resultsSet.resultValues[self.index]

    def set_values(self, values):
        self.resultsSet.resultValues[self.index] = values


class ResultsSet:
    def __init__(self, label=None, timestamp=None):
        self.label = label
        self.timestamp = timestamp or datetime.datetime.utcnow()
        self.resultValues = []

    def count(self):
        return len(self.resultValues)

    def add(self, results):
        self.resultValues.append(results.tolist())

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.resultValues):
            element = ResultsElement(self, self.index)
            self.index += 1
            return element
        else:
            raise StopIteration

    def encode(self):
        items = []
        for item in self:
            items.append({'index':item.index+1,
                          'values':item.get_values()
                          })

        return { '__ResultsSet__':True, 'label':self.label, 'createdAt':str(self.timestamp), 'count': self.count(), 'items':items }

    def decode(self, itemsArray):
        for item in itemsArray:
            self.resultValues.append(as_ndarray(item["values"]))