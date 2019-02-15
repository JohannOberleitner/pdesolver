import abc


class LexerToken(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def precedence(self):
        """Return token precedence"""


class DummyLexerToken(LexerToken):
    def precedence(self):
        return -1


class NumberToken(LexerToken):

    def __init__(self, value):
        self.stringValue = value

    def precedence(self):
        return 1

    @staticmethod
    def matches(c):
        return c == '.' or c.isdigit()

    def get_value(self):
        return float(self.stringValue)

    def __repr__(self):
        return 'Number:{0}'.format(float(self.stringValue))


class LeftBraceToken(LexerToken):

    def precedence(self):
        return 2

    @staticmethod
    def matches(c):
        return c == '('

    def __repr__(self):
        return 'LBRACE'


class RightBraceToken(LexerToken):

    def precedence(self):
            return 2

    @staticmethod
    def matches(c):
        return c == ')'

    def __repr__(self):
        return 'RBRACE'


class LeftCurlyBraceToken(LexerToken):

    def precedence(self):
        return 2

    @staticmethod
    def matches(c):
        return c == '{'

    def __repr__(self):
        return 'LCBRACE'


class RightCurlyBraceToken(LexerToken):

    def precedence(self):
        return 2

    @staticmethod
    def matches(c):
        return c == '}'

    def __repr__(self):
        return 'RCBRACE'


class BinaryOperatorToken(LexerToken):

    def __init__(self, symbol):
        self.symbol = symbol

    @staticmethod
    def matches(c):
        return c in '+-*/'

    def get_symbol(self):
        return self.symbol

    def precedence(self):

        if self.symbol in '*/':
            return 4
        elif self.symbol in '+-':
            return 3
        else:
            raise Exception('Invalid symbol: {0}'.format(self.symbol))

    def __repr__(self):
        return 'BinaryOp: {0}'.format(self.get_symbol())


class UnaryOperatorToken(LexerToken):

    def __init__(self, symbol):
        self.symbol = symbol

    @staticmethod
    def matches(c):
        return c in '-'

    def get_symbol(self):
            return self.symbol

    def precedence(self):
        return 2

    def __repr__(self):
        return 'UnaryOp: {0}'.format(self.get_symbol())


class VariableNameToken(LexerToken):

    def __init__(self, name):
        self.name = name

    @staticmethod
    def matches(c):
        return c == '_' or c.isalnum()

    def get_name(self):
        return self.name

    def precedence(self):
        return 1

    def __repr__(self):
        return 'Variable: {0}'.format(self.get_name())


class FunctionCallToken(LexerToken):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def matches(c):
        return c == '_' or c.isalnum()

    def get_name(self):
        return self.name

    def precedence(self):
            return 1

    def __repr__(self):
        return 'Function call: {0}'.format(self.get_name())

class ParameterSeparatorToken(LexerToken):

    def precedence(self):
        return 5

    @staticmethod
    def matches(c):
        return c == ','

    def __repr__(self):
        return 'PARAMSEP'