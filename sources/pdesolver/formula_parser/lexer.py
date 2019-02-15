from sources.pdesolver.formula_parser.lexer_token import *


class State:
    Start = 0
    Number = 1
    VariableName = 2,
    CurlySeparatedVariableName = 3


class Lexer:

    def __init__(self, input_text):
        self.input = input_text
        self.state = State.Start
        self.last_lexem = []

    def is_number_token_possible(self):
        return self.state == State.Start or self.state == State.Number

    def is_variable_token_possible(self):
        return self.state == State.Start or \
               self.state == State.VariableName or\
               self.state == State.CurlySeparatedVariableName

    def is_curly_brace_variable_token_possible(self):
        return self.state == State.CurlySeparatedVariableName

    def process_lastlexem(self):
        token = None
        if len(self.last_lexem) > 0:

            if self.state == State.VariableName:

                token = VariableNameToken(''.join(self.last_lexem))
                self.last_lexem.clear()
                self.state = State.Start

            elif self.state == State.Number:
                token = NumberToken(''.join(self.last_lexem))
                self.last_lexem.clear()
                self.state = State.Start

        return token

    def parse(self):
        previous_token = None

        for c in self.input:

            # if WhiteSpace in variable names: continue
            if self.state != State.VariableName and self.state != State.CurlySeparatedVariableName and c.isspace():
                continue

            # continue if not in a CurlySeparatedVariable name and any non-number of
            # non-variable characters are appearing
            if self.state != State.CurlySeparatedVariableName and (
                LeftBraceToken.matches(c)
                or RightBraceToken.matches(c)
                or UnaryOperatorToken.matches(c)
                or BinaryOperatorToken.matches(c)
                or ParameterSeparatorToken.matches(c)
            ):
                completed_token = self.process_lastlexem()
                if completed_token is not None:
                    previous_token = completed_token
                    yield completed_token

                if LeftBraceToken.matches(c):
                    next_token = LeftBraceToken()
                    previous_token = next_token
                    yield next_token
                elif RightBraceToken.matches(c):
                    next_token = RightBraceToken()
                    previous_token = next_token
                    yield next_token
                elif ParameterSeparatorToken.matches(c):
                    next_token = ParameterSeparatorToken()
                    previous_token = next_token
                    yield next_token
                elif UnaryOperatorToken.matches(c) and (previous_token is None or type(previous_token) is LeftBraceToken):
                    next_token = UnaryOperatorToken(c)
                    previous_token = next_token
                    yield next_token
                elif BinaryOperatorToken.matches(c):
                    next_token = BinaryOperatorToken(c)
                    previous_token = next_token
                    yield next_token
            elif self.is_number_token_possible() and NumberToken.matches(c):
                self.state = State.Number
                self.last_lexem.append(c)
            elif self.is_variable_token_possible() and VariableNameToken.matches(c):
                if self.state != State.CurlySeparatedVariableName:
                    self.state = State.VariableName
                    self.last_lexem.append(c)
            elif self.is_curly_brace_variable_token_possible() \
                    and not LeftCurlyBraceToken.matches(c) \
                    and not RightCurlyBraceToken.matches(c):
                self.last_lexem.append(c)
            elif LeftCurlyBraceToken.matches(c):
                self.state = State.CurlySeparatedVariableName
            elif RightCurlyBraceToken.matches(c):
                next_token = VariableNameToken(self.last_lexem)
                self.last_lexem.clear()
                self.state = State.Start
                previous_token = next_token
                yield next_token

        previous_token = self.process_lastlexem()
        if previous_token is not None:
            yield previous_token
