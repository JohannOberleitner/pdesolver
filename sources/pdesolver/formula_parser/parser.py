from sources.pdesolver.formula_parser.lexer_token import *
from sources.pdesolver.formula_parser.parser_expression import *


class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.stack = []
        self.last_tokens = []
        self.token_index = 0

    def peek_next_token(self):
        if self.token_index < len(self.tokens):
            return self.tokens[self.token_index], type(self.tokens[self.token_index])
        else:
            return None,None

    def parse(self):
        return self.parse_internal(DummyLexerToken())


    def parse_internal(self, previous_token):
        while self.token_index < len(self.tokens):
            token = self.tokens[self.token_index]
            token_type = type(token)
            self.token_index += 1

            if token_type is NumberToken:
                expr = NumberExpression(token.get_value())
                self.stack.append(expr)
            elif token_type is VariableNameToken:
                next, next_type = self.peek_next_token()
                if next_type is LeftBraceToken:
                    self.token_index += 1
                    parameter_expr_list = []
                    next, next_type = self.peek_next_token()
                    while not next_type is RightBraceToken:

                        try:
                            param_expr = self.parse_internal(next)

                            parameter_expr_list.append(param_expr)

                            self.token_index -= 1  # step back
                            next, next_type = self.peek_next_token()
                            if next_type is ParameterSeparatorToken:
                                self.token_index += 1 # step forward
                                next, next_type = self.peek_next_token()
                        except Exception as iexcpt:
                            print(iexcpt)
                    self.token_index += 1
                    expr = FunctionCallExpression(token.get_name(),parameter_expr_list)
                else:
                    expr = VariableExpression(token.get_name())
                self.stack.append(expr)
            elif token_type is LeftBraceToken:
                expr = self.parse_internal(DummyLexerToken())
                self.stack.append(InnerExpression(expr))
            elif token_type is RightBraceToken:
                expr = self.stack.pop()
                return expr
            elif token_type is ParameterSeparatorToken:
                expr = self.stack.pop()
                return expr
            elif token_type is UnaryOperatorToken:
                symbol = token.get_symbol()
                child = self.parse_internal(token)
                expr = UnaryOperatorExpression(child, symbol)
                self.stack.append(expr)
            elif token_type is BinaryOperatorToken:
                symbol = token.get_symbol()
                right_child = self.parse_internal(token)
                left_child = self.stack.pop()
                expr = BinaryOperatorExpression(left_child, right_child, symbol)
                self.stack.append(expr)

            next_token,_ = self.peek_next_token()

            if next_token is not None\
                    and previous_token is not None\
                    and next_token.precedence() <= previous_token.precedence():
                return self.stack.pop()

        return self.stack.pop()
