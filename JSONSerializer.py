import collections
import json

class JSONSerializer(object):

    def __init__(self, identifier: str, parameters = None):
        self.identifier = identifier
        if (parameters is None):
            self.parameters = collections.defaultdict()
        else:
            self.parameters = parameters


    @classmethod
    def fromJSON(cls, jsonString: str):
        data = json.loads(jsonString)
        return cls(data['identifier'], parameters=data['parameters'])


    def toJSON(self):
        data = {
            'identifier': self.identifier,
            'parameters': {}
        }
        for key, value in self.parameters.items():
            if (not isinstance(value, dict)):
                data['parameters'][key] = value
            else:
                data['parameters']['parameters'] = value
        return data