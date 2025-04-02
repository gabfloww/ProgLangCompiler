
import re
import sys
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union


class TokenType(Enum):
    # Keywords
    VAR = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    PRINT = auto()
    FUNC = auto()
    RETURN = auto()
    
    # Types
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    IDENTIFIER = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()
    
    # Comparison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    
    # Special
    EOF = auto()
    ERROR = auto()

class Token:
    def __init__(self, token_type: TokenType, value: str, line: int, column: int):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
    
    def __str__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"

class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if len(self.source) > 0 else None
        
        # Keywords
        self.keywords = {
            'var': TokenType.VAR,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'print': TokenType.PRINT,
            'func': TokenType.FUNC,
            'return': TokenType.RETURN,
            'int': TokenType.INT,
            'float': TokenType.FLOAT,
            'string': TokenType.STRING
        }
    
    def advance(self):
        """Move to the next character in source code."""
        self.position += 1
        self.column += 1
        
        if self.position >= len(self.source):
            self.current_char = None
        else:
            self.current_char = self.source[self.position]
            
            # Handle newlines for line counting
            if self.current_char == '\n':
                self.line += 1
                self.column = 0
    
    def peek(self, n=1):
        """Look ahead n characters without advancing."""
        peek_pos = self.position + n
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments."""
        if self.current_char == '/' and self.peek() == '/':
            # Skip to the end of the line
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()
    
    def read_identifier(self):
        """Read an identifier or keyword."""
        start_col = self.column
        id_str = ""
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            id_str += self.current_char
            self.advance()
        
        # Check if it's a keyword or an identifier
        token_type = self.keywords.get(id_str, TokenType.IDENTIFIER)
        return Token(token_type, id_str, self.line, start_col)
    
    def read_number(self):
        """Read a number (integer or float)."""
        start_col = self.column
        num_str = ""
        is_float = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if is_float:  # Second decimal point is an error
                    return Token(TokenType.ERROR, "Invalid number format", self.line, start_col)
                is_float = True
            
            num_str += self.current_char
            self.advance()
        
        if is_float:
            return Token(TokenType.FLOAT_LIT, num_str, self.line, start_col)
        else:
            return Token(TokenType.INT_LIT, num_str, self.line, start_col)
    
    def read_string(self):
        """Read a string literal."""
        start_col = self.column
        self.advance()  # Skip opening quote
        string_value = ""
        
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\' and self.peek() == '"':
                # Handle escaped quotes
                string_value += '"'
                self.advance()  # Skip backslash
                self.advance()  # Skip quote
            else:
                string_value += self.current_char
                self.advance()
        
        if self.current_char is None:
            return Token(TokenType.ERROR, "Unterminated string", self.line, start_col)
        
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING_LIT, string_value, self.line, start_col)
    
    def get_next_token(self):
        """Get the next token from source code."""
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and self.peek() == '/':
                self.skip_comment()
                continue
            
            # Identifier or keyword
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()
            
            # Number
            if self.current_char.isdigit():
                return self.read_number()
            
            # String literal
            if self.current_char == '"':
                return self.read_string()
            
            # Operators and delimiters
            if self.current_char == '+':
                token = Token(TokenType.PLUS, '+', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '-':
                token = Token(TokenType.MINUS, '-', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '*':
                token = Token(TokenType.MULTIPLY, '*', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '/':
                token = Token(TokenType.DIVIDE, '/', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '=':
                if self.peek() == '=':
                    col = self.column
                    self.advance()  # Skip first =
                    self.advance()  # Skip second =
                    return Token(TokenType.EQUAL, '==', self.line, col)
                else:
                    token = Token(TokenType.ASSIGN, '=', self.line, self.column)
                    self.advance()
                    return token
            
            if self.current_char == '!':
                if self.peek() == '=':
                    col = self.column
                    self.advance()  # Skip !
                    self.advance()  # Skip =
                    return Token(TokenType.NOT_EQUAL, '!=', self.line, col)
            
            if self.current_char == '<':
                if self.peek() == '=':
                    col = self.column
                    self.advance()  # Skip <
                    self.advance()  # Skip =
                    return Token(TokenType.LESS_EQUAL, '<=', self.line, col)
                else:
                    token = Token(TokenType.LESS, '<', self.line, self.column)
                    self.advance()
                    return token
            
            if self.current_char == '>':
                if self.peek() == '=':
                    col = self.column
                    self.advance()  # Skip >
                    self.advance()  # Skip =
                    return Token(TokenType.GREATER_EQUAL, '>=', self.line, col)
                else:
                    token = Token(TokenType.GREATER, '>', self.line, self.column)
                    self.advance()
                    return token
            
            if self.current_char == '(':
                token = Token(TokenType.LPAREN, '(', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == ')':
                token = Token(TokenType.RPAREN, ')', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '{':
                token = Token(TokenType.LBRACE, '{', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == '}':
                token = Token(TokenType.RBRACE, '}', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == ';':
                token = Token(TokenType.SEMICOLON, ';', self.line, self.column)
                self.advance()
                return token
            
            if self.current_char == ',':
                token = Token(TokenType.COMMA, ',', self.line, self.column)
                self.advance()
                return token
            
            # If we get here, we found an invalid character
            token = Token(TokenType.ERROR, f"Invalid character: '{self.current_char}'", self.line, self.column)
            self.advance()
            return token
        
        # End of file
        return Token(TokenType.EOF, '', self.line, self.column)

    def tokenize(self):
        """Convert source code to a list of tokens."""
        tokens = []
        token = self.get_next_token()
        
        while token.type != TokenType.EOF:
            if token.type == TokenType.ERROR:
                raise SyntaxError(f"Lexical error at line {token.line}, column {token.column}: {token.value}")
            tokens.append(token)
            token = self.get_next_token()
        
        tokens.append(token)  # Add EOF token
        return tokens

#################################################
# 2. SYNTAX ANALYZER (PARSER)
#################################################

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class VarDeclaration(ASTNode):
    def __init__(self, var_type, name, initial_value=None):
        self.var_type = var_type
        self.name = name
        self.initial_value = initial_value

class FunctionDecl(ASTNode):
    def __init__(self, return_type, name, params, body):
        self.return_type = return_type
        self.name = name
        self.params = params  # List of (type, name) tuples
        self.body = body

class ReturnStmt(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class Parameter(ASTNode):
    def __init__(self, param_type, name):
        self.param_type = param_type
        self.name = name

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class AssignmentStmt(ASTNode):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class IfStmt(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

class WhileStmt(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStmt(ASTNode):
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

class PrintStmt(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BinaryExpr(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class UnaryExpr(ASTNode):
    def __init__(self, operator, expr):
        self.operator = operator
        self.expr = expr

class FunctionCall(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class IntLiteral(ASTNode):
    def __init__(self, value):
        self.value = int(value)

class FloatLiteral(ASTNode):
    def __init__(self, value):
        self.value = float(value)

class StringLiteral(ASTNode):
    def __init__(self, value):
        self.value = value

class Identifier(ASTNode):
    def __init__(self, name):
        self.name = name

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
    
    def current_token(self):
        """Get the current token."""
        return self.tokens[self.current]
    
    def peek(self):
        """Look at the next token without consuming it."""
        if self.current + 1 >= len(self.tokens):
            return self.tokens[-1]  # Return EOF token
        return self.tokens[self.current + 1]
    
    def check(self, token_type):
        """Check if current token is of specified type."""
        return self.current_token().type == token_type
    
    def match(self, *token_types):
        """Match current token against specified types."""
        for token_type in token_types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def consume(self, token_type, error_msg):
        """Consume current token if it matches the type, otherwise throw error."""
        if self.check(token_type):
            return self.advance()
        
        token = self.current_token()
        raise SyntaxError(f"Parse error at line {token.line}, column {token.column}: {error_msg}")
    
    def advance(self):
        """Advance to the next token."""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.current += 1
        return token
    
    def parse(self):
        """Parse the tokens into an AST."""
        statements = []
        
        while not self.check(TokenType.EOF):
            statements.append(self.parse_statement())
        
        return Program(statements)
    
    def parse_statement(self):
        """Parse a statement."""
        if self.match(TokenType.VAR):
            return self.parse_var_declaration()
        elif self.match(TokenType.FUNC):
            return self.parse_function_declaration()
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.FOR):
            return self.parse_for_statement()
        elif self.match(TokenType.PRINT):
            return self.parse_print_statement()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.LBRACE):
            return self.parse_block()
        else:
            # Must be expression statement or assignment
            return self.parse_expression_or_assignment()
    
    def parse_var_declaration(self):
        """Parse a variable declaration."""
        # var_type already consumed with the 'var' keyword
        var_type_token = self.consume(TokenType.INT, TokenType.FLOAT, TokenType.STRING, "Expected variable type")
        var_type = var_type_token.value
        
        name_token = self.consume(TokenType.IDENTIFIER, "Expected variable name")
        name = name_token.value
        
        initial_value = None
        if self.match(TokenType.ASSIGN):
            initial_value = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        return VarDeclaration(var_type, name, initial_value)
    
    def parse_function_declaration(self):
        """Parse a function declaration."""
        # 'func' keyword already consumed
        return_type_token = self.consume(TokenType.INT, TokenType.FLOAT, TokenType.STRING, "Expected return type")
        return_type = return_type_token.value
        
        name_token = self.consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        
        # Parse parameters
        params = []
        if not self.check(TokenType.RPAREN):
            params.append(self.parse_parameter())
            
            while self.match(TokenType.COMMA):
                params.append(self.parse_parameter())
        
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        # Parse function body
        body = self.parse_block()
        
        return FunctionDecl(return_type, name, params, body)
    
    def parse_parameter(self):
        """Parse a function parameter."""
        type_token = self.consume(TokenType.INT, TokenType.FLOAT, TokenType.STRING, "Expected parameter type")
        param_type = type_token.value
        
        name_token = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
        name = name_token.value
        
        return Parameter(param_type, name)
    
    def parse_block(self):
        """Parse a block of statements."""
        # '{' already consumed
        statements = []
        
        while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
            statements.append(self.parse_statement())
        
        self.consume(TokenType.RBRACE, "Expected '}' after block")
        return Block(statements)
    
    def parse_if_statement(self):
        """Parse an if statement."""
        # 'if' already consumed
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        
        if_body = self.parse_statement()
        
        else_body = None
        if self.match(TokenType.ELSE):
            else_body = self.parse_statement()
        
        return IfStmt(condition, if_body, else_body)
    
    def parse_while_statement(self):
        """Parse a while statement."""
        # 'while' already consumed
        self.consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        
        body = self.parse_statement()
        
        return WhileStmt(condition, body)
    
    def parse_for_statement(self):
        """Parse a for statement."""
        # 'for' already consumed
        self.consume(TokenType.LPAREN, "Expected '(' after 'for'")
        
        # Initialization
        init = None
        if not self.check(TokenType.SEMICOLON):
            if self.check(TokenType.VAR):
                self.advance()  # Consume 'var'
                init = self.parse_var_declaration()
            else:
                init = self.parse_expression_or_assignment()
                self.consume(TokenType.SEMICOLON, "Expected ';' after for loop initialization")
        else:
            self.advance()  # Consume ';'
        
        # Condition
        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for loop condition")
        
        # Update
        update = None
        if not self.check(TokenType.RPAREN):
            update = self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after for loop clauses")
        
        # Body
        body = self.parse_statement()
        
        return ForStmt(init, condition, update, body)
    
    def parse_print_statement(self):
        """Parse a print statement."""
        # 'print' already consumed
        expr = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after print statement")
        return PrintStmt(expr)
    
    def parse_return_statement(self):
        """Parse a return statement."""
        # 'return' already consumed
        expr = None
        if not self.check(TokenType.SEMICOLON):
            expr = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return statement")
        return ReturnStmt(expr)
    
    def parse_expression_or_assignment(self):
        """Parse an expression or assignment statement."""
        expr = self.parse_expression()
        
        if isinstance(expr, Identifier) and self.match(TokenType.ASSIGN):
            value = self.parse_expression()
            self.consume(TokenType.SEMICOLON, "Expected ';' after assignment")
            return AssignmentStmt(expr.name, value)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return expr
    
    def parse_expression(self):
        """Parse an expression."""
        return self.parse_equality()
    
    def parse_equality(self):
        """Parse an equality expression."""
        expr = self.parse_comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.tokens[self.current - 1].type
            right = self.parse_comparison()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def parse_comparison(self):
        """Parse a comparison expression."""
        expr = self.parse_term()
        
        while self.match(TokenType.LESS, TokenType.GREATER, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL):
            operator = self.tokens[self.current - 1].type
            right = self.parse_term()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def parse_term(self):
        """Parse a term."""
        expr = self.parse_factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.tokens[self.current - 1].type
            right = self.parse_factor()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def parse_factor(self):
        """Parse a factor."""
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE):
            operator = self.tokens[self.current - 1].type
            right = self.parse_unary()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def parse_unary(self):
        """Parse a unary expression."""
        if self.match(TokenType.MINUS):
            operator = self.tokens[self.current - 1].type
            right = self.parse_unary()
            return UnaryExpr(operator, right)
        
        return self.parse_primary()
    
    def parse_primary(self):
        """Parse a primary expression."""
        if self.match(TokenType.INT_LIT):
            value = self.tokens[self.current - 1].value
            return IntLiteral(value)
        
        if self.match(TokenType.FLOAT_LIT):
            value = self.tokens[self.current - 1].value
            return FloatLiteral(value)
        
        if self.match(TokenType.STRING_LIT):
            value = self.tokens[self.current - 1].value
            return StringLiteral(value)
        
        if self.match(TokenType.IDENTIFIER):
            name = self.tokens[self.current - 1].value
            
            # Check if it's a function call
            if self.match(TokenType.LPAREN):
                args = []
                
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())
                
                self.consume(TokenType.RPAREN, "Expected ')' after arguments")
                return FunctionCall(name, args)
            
            return Identifier(name)
        
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        token = self.current_token()
        raise SyntaxError(f"Unexpected token at line {token.line}, column {token.column}: {token.value}")

#################################################
# 3. SEMANTIC ANALYZER
#################################################

class Symbol:
    def __init__(self, name, symbol_type, scope="global"):
        self.name = name
        self.type = symbol_type
        self.scope = scope

class Function(Symbol):
    def __init__(self, name, return_type, params, scope="global"):
        super().__init__(name, "function", scope)
        self.return_type = return_type
        self.params = params  # List of (type, name) tuples

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.scopes = ["global"]
        self.current_scope = "global"
    
    def enter_scope(self, scope_name=None):
        """Enter a new scope."""
        if scope_name is None:
            scope_name = f"{self.current_scope}.{len(self.scopes)}"
        self.scopes.append(scope_name)
        self.current_scope = scope_name
    
    def exit_scope(self):
        """Exit the current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
            self.current_scope = self.scopes[-1]
    
    def add_symbol(self, symbol):
        """Add a symbol to the current scope."""
        key = f"{self.current_scope}.{symbol.name}"
        if key in self.symbols:
            return False  # Symbol already exists in this scope
        self.symbols[key] = symbol
        return True
    
    def lookup(self, name, current_scope_only=False):
        """Look up a symbol by name."""
        # Try local scope first
        key = f"{self.current_scope}.{name}"
        if key in self.symbols:
            return self.symbols[key]
        
        if current_scope_only:
            return None
        
        # Try parent scopes
        scope_parts = self.current_scope.split('.')
        while len(scope_parts) > 0:
            scope_parts.pop()
            parent_scope = '.'.join(scope_parts) if scope_parts else "global"
            key = f"{parent_scope}.{name}"
            if key in self.symbols:
                return self.symbols[key]
        
        # Finally, try global scope
        key = f"global.{name}"
        if key in self.symbols:
            return self.symbols[key]
        
        return None

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []
        self.current_function = None
    
    def analyze(self, ast):
        """Analyze the abstract syntax tree."""
        self.visit(ast)
        return len(self.errors) == 0, self.errors
    
    def visit(self, node):
        """Visit a node in the AST."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Default handler for node types without specific visitors."""
        return None
    
    def visit_Program(self, node):
        """Visit a program node."""
        for statement in node.statements:
            self.visit(statement)
    
    def visit_VarDeclaration(self, node):
        """Visit a variable declaration."""
        # Check if variable already exists in current scope
        if self.symbol_table.lookup(node.name, True):
            self.errors.append(f"Redeclaration of variable '{node.name}'")
            return
        
        # Add variable to symbol table
        symbol = Symbol(node.name, node.var_type, self.symbol_table.current_scope)
        self.symbol_table.add_symbol(symbol)
        
        # Check initialization if present
        if node.initial_value:
            init_type = self.visit(node.initial_value)
            if init_type != node.var_type and not (node.var_type == "float" and init_type == "int"):
                self.errors.append(f"Type mismatch in variable '{node.name}' initialization: expected {node.var_type}, got {init_type}")
    
    def visit_FunctionDecl(self, node):
        """Visit a function declaration."""
        # Check if function already exists
        if self.symbol_table.lookup(node.name, True):
            self.errors.append(f"Redeclaration of function '{node.name}'")
            return
        
        # Add function to symbol table
        params = [(p.param_type, p.name) for p in node.params]
        function = Function(node.name, node.return_type, params)
        self.symbol_table.add_symbol(function)
        
        # Save current function for return type checking
        old_function = self.current_function
        self.current_function = function
        
        # Enter function scope
        self.symbol_table.enter_scope(f"function.{node.name}")
        
        # Add parameters to symbol table
        for param in node.params:
            param_symbol = Symbol(param.name, param.param_type, self.symbol_table.current_scope)
            self.symbol_table.add_symbol(param_symbol)
        
        # Check function body
        self.visit(node.body)
        
        # Exit function scope
        self.symbol_table.exit_scope()
        
        # Restore previous function
        self.current_function = old_function
    
    def visit_Block(self, node):
        """Visit a block of statements."""
        self.symbol_table.enter_scope()
        
        for statement in node.statements:
            self.visit(statement)
        
        self.symbol_table.exit_scope()
    
    def visit_ReturnStmt(self, node):
        """Visit a return statement."""
        if not self.current_function:
            self.errors.append("Return statement outside of function")
            return
        
        # Check if return type matches function return type
        if node.expr:
            expr_type = self.visit(node.expr)
            if expr_type != self.current_function.return_type and not (self.current_function.return_type == "float" and expr_type == "int"):
                self.errors.append(f"Return type mismatch: expected {self.current_function.return_type}, got {expr_type}")
        elif self.current_function.return_type != "void":
            self.errors.append(f"Function '{self.current_function.name}' must return a value of type {self.current_function.return_type}")
    
    def visit_AssignmentStmt(self, node):
        """Visit an assignment statement."""
        # Check if variable exists
        var = self.symbol_table.lookup(node.name)
        if not var:
            self.errors.append(f"Undefined variable '{node.name}'")
            return
        
        # Check if types match
        expr_type = self.visit(node.expr)
        if expr_type != var.type and not (var.type == "float" and expr_type == "int"):
            self.errors.append(f"Type mismatch in assignment to '{node.name}': expected {var.type}, got {expr_type}")
    
    def visit_IfStmt(self, node):
        """Visit an if statement."""
        # Check condition type
        cond_type = self.visit(node.condition)
        if cond_type not in ["int", "float"]:  # Treating non-zero as true
            self.errors.append(f"Condition must be a numeric type, got {cond_type}")
        
        # Check if and else bodies
        self.visit(node.if_body)
        if node.else_body:
            self.visit(node.else_body)
    
    def visit_WhileStmt(self, node):
        """Visit a while statement."""
        # Check condition type
        cond_type = self.visit(node.condition)
        if cond_type not in ["int", "float"]:  # Treating non-zero as true
            self.errors.append(f"Condition must be a numeric type, got {cond_type}")
        
        # Check body
        self.visit(node.body)
    
    def visit_ForStmt(self, node):
        """Visit a for statement."""
        # Enter loop scope
        self.symbol_table.enter_scope()
        
        # Check initialization
        if node.init:
            self.visit(node.init)
        
        # Check condition
        if node.condition:
            cond_type = self.visit(node.condition)
            if cond_type not in ["int", "float"]:
                self.errors.append(f"For loop condition must be a numeric type, got {cond_type}")
        
        # Check update
        if node.update:
            self.visit(node.update)
        
        # Check body
        self.visit(node.body)
        
        # Exit loop scope
        self.symbol_table.exit_scope()
    
    def visit_PrintStmt(self, node):
        """Visit a print statement."""
        # Any type can be printed, so just visit the expression
        self.visit(node.expr)
    
    def visit_BinaryExpr(self, node):
        """Visit a binary expression."""
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        # Comparison operators
        if node.operator in [TokenType.EQUAL, TokenType.NOT_EQUAL, 
                            TokenType.LESS, TokenType.GREATER, 
                            TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL]:
            if left_type not in ["int", "float"] or right_type not in ["int", "float"]:
                self.errors.append(f"Comparison operators require numeric operands, got {left_type} and {right_type}")
            return "int"  # Boolean result is represented as int
        
        # Arithmetic operators
        if left_type not in ["int", "float"] or right_type not in ["int", "float"]:
            self.errors.append(f"Arithmetic operators require numeric operands, got {left_type} and {right_type}")
        
        # Result type is float if either operand is float
        if left_type == "float" or right_type == "float":
            return "float"
        return "int"
    
    def visit_UnaryExpr(self, node):
        """Visit a unary expression."""
        expr_type = self.visit(node.expr)
        
        if node.operator == TokenType.MINUS:
            if expr_type not in ["int", "float"]:
                self.errors.append(f"Unary minus requires numeric operand, got {expr_type}")
            return expr_type
        
        return expr_type
    
    def visit_FunctionCall(self, node):
        """Visit a function call."""
        # Check if function exists
        func = self.symbol_table.lookup(node.name)
        if not func or func.type != "function":
            self.errors.append(f"Undefined function '{node.name}'")
            return "unknown"
        
        # Check number of arguments
        if len(node.args) != len(func.params):
            self.errors.append(f"Function '{node.name}' expects {len(func.params)} arguments, got {len(node.args)}")
        else:
            # Check argument types
            for i, (arg, param) in enumerate(zip(node.args, func.params)):
                arg_type = self.visit(arg)
                param_type = param[0]  # param is a (type, name) tuple
                if arg_type != param_type and not (param_type == "float" and arg_type == "int"):
                    self.errors.append(f"Argument {i+1} of function '{node.name}' has wrong type: expected {param_type}, got {arg_type}")
        
        return func.return_type
    
    def visit_IntLiteral(self, node):
        """Visit an integer literal."""
        return "int"
    
    def visit_FloatLiteral(self, node):
        """Visit a float literal."""
        return "float"
    
    def visit_StringLiteral(self, node):
        """Visit a string literal."""
        return "string"
    
    def visit_Identifier(self, node):
        """Visit an identifier."""
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self.errors.append(f"Undefined variable '{node.name}'")
            return "unknown"
        return symbol.type

#################################################
# 4. INTERMEDIATE REPRESENTATION (IR) GENERATOR
#################################################

class IRType(Enum):
    DECLARE = auto()
    ASSIGN = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    JUMP = auto()
    JUMP_IF_TRUE = auto()
    JUMP_IF_FALSE = auto()
    LABEL = auto()
    CALL = auto()
    RETURN = auto()
    PARAM = auto()
    PRINT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

class IRInstruction:
    def __init__(self, op_type, result=None, arg1=None, arg2=None):
        self.op_type = op_type
        self.result = result
        self.arg1 = arg1
        self.arg2 = arg2
    
    def __str__(self):
        if self.op_type == IRType.DECLARE:
            return f"DECLARE {self.result} {self.arg1}"
        elif self.op_type == IRType.ASSIGN:
            return f"{self.result} = {self.arg1}"
        elif self.op_type == IRType.ADD:
            return f"{self.result} = {self.arg1} + {self.arg2}"
        elif self.op_type == IRType.SUB:
            return f"{self.result} = {self.arg1} - {self.arg2}"
        elif self.op_type == IRType.MUL:
            return f"{self.result} = {self.arg1} * {self.arg2}"
        elif self.op_type == IRType.DIV:
            return f"{self.result} = {self.arg1} / {self.arg2}"
        elif self.op_type == IRType.JUMP:
            return f"JUMP {self.result}"
        elif self.op_type == IRType.JUMP_IF_TRUE:
            return f"JUMP_IF_TRUE {self.arg1} {self.result}"
        elif self.op_type == IRType.JUMP_IF_FALSE:
            return f"JUMP_IF_FALSE {self.arg1} {self.result}"
        elif self.op_type == IRType.LABEL:
            return f"LABEL {self.result}"
        elif self.op_type == IRType.CALL:
            return f"{self.result} = CALL {self.arg1}({self.arg2})"
        elif self.op_type == IRType.RETURN:
            return f"RETURN {self.result}" if self.result else "RETURN"
        elif self.op_type == IRType.PARAM:
            return f"PARAM {self.result}"
        elif self.op_type == IRType.PRINT:
            return f"PRINT {self.result}"
        elif self.op_type == IRType.EQ:
            return f"{self.result} = {self.arg1} == {self.arg2}"
        elif self.op_type == IRType.NE:
            return f"{self.result} = {self.arg1} != {self.arg2}"
        elif self.op_type == IRType.LT:
            return f"{self.result} = {self.arg1} < {self.arg2}"
        elif self.op_type == IRType.GT:
            return f"{self.result} = {self.arg1} > {self.arg2}"
        elif self.op_type == IRType.LE:
            return f"{self.result} = {self.arg1} <= {self.arg2}"
        elif self.op_type == IRType.GE:
            return f"{self.result} = {self.arg1} >= {self.arg2}"
        else:
            return f"{self.op_type} {self.result} {self.arg1} {self.arg2}"

class IRGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
        self.current_function = None
    
    def generate(self, ast):
        """Generate IR code from the AST."""
        self.visit(ast)
        return self.instructions
    
    def visit(self, node):
        """Visit a node in the AST."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Default handler for node types without specific visitors."""
        return None
    
    def new_temp(self):
        """Generate a new temporary variable name."""
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self):
        """Generate a new label."""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def visit_Program(self, node):
        """Visit a program node."""
        for statement in node.statements:
            self.visit(statement)
    
    def visit_VarDeclaration(self, node):
        """Visit a variable declaration."""
        self.instructions.append(IRInstruction(IRType.DECLARE, node.name, node.var_type))
        
        if node.initial_value:
            value = self.visit(node.initial_value)
            self.instructions.append(IRInstruction(IRType.ASSIGN, node.name, value))
    
    def visit_FunctionDecl(self, node):
        """Visit a function declaration."""
        # Store current function
        old_function = self.current_function
        self.current_function = node.name
        
        # Function label
        self.instructions.append(IRInstruction(IRType.LABEL, f"function_{node.name}"))
        
        # Declare parameters
        for param in node.params:
            self.instructions.append(IRInstruction(IRType.DECLARE, param.name, param.param_type))
        
        # Function body
        self.visit(node.body)
        
        # Implicit return for void functions
        if not any(instr.op_type == IRType.RETURN for instr in self.instructions[-len(node.body.statements):]):
            self.instructions.append(IRInstruction(IRType.RETURN))
        
        # Restore previous function
        self.current_function = old_function
    
    def visit_ReturnStmt(self, node):
        """Visit a return statement."""
        if node.expr:
            value = self.visit(node.expr)
            self.instructions.append(IRInstruction(IRType.RETURN, value))
        else:
            self.instructions.append(IRInstruction(IRType.RETURN))
    
    def visit_Block(self, node):
        """Visit a block of statements."""
        for statement in node.statements:
            self.visit(statement)
    
    def visit_AssignmentStmt(self, node):
        """Visit an assignment statement."""
        value = self.visit(node.expr)
        self.instructions.append(IRInstruction(IRType.ASSIGN, node.name, value))
        return node.name
    
    def visit_IfStmt(self, node):
        """Visit an if statement."""
        condition = self.visit(node.condition)
        
        if node.else_body:
            else_label = self.new_label()
            end_label = self.new_label()
            
            # Jump to else if condition is false
            self.instructions.append(IRInstruction(IRType.JUMP_IF_FALSE, else_label, condition))
            
            # If body
            self.visit(node.if_body)
            self.instructions.append(IRInstruction(IRType.JUMP, end_label))
            
            # Else body
            self.instructions.append(IRInstruction(IRType.LABEL, else_label))
            self.visit(node.else_body)
            
            # End of if-else
            self.instructions.append(IRInstruction(IRType.LABEL, end_label))
        else:
            # Just if without else
            end_label = self.new_label()
            self.instructions.append(IRInstruction(IRType.JUMP_IF_FALSE, end_label, condition))
            self.visit(node.if_body)
            self.instructions.append(IRInstruction(IRType.LABEL, end_label))
    
    def visit_WhileStmt(self, node):
        """Visit a while statement."""
        start_label = self.new_label()
        end_label = self.new_label()
        
        # Loop start
        self.instructions.append(IRInstruction(IRType.LABEL, start_label))
        
        # Condition check
        condition = self.visit(node.condition)
        self.instructions.append(IRInstruction(IRType.JUMP_IF_FALSE, end_label, condition))
        
        # Body
        self.visit(node.body)
        
        # Jump back to start
        self.instructions.append(IRInstruction(IRType.JUMP, start_label))
        
        # Loop end
        self.instructions.append(IRInstruction(IRType.LABEL, end_label))
    
    def visit_ForStmt(self, node):
        """Visit a for statement."""
        start_label = self.new_label()
        update_label = self.new_label()
        end_label = self.new_label()
        
        # Initialization
        if node.init:
            self.visit(node.init)
        
        # Loop start
        self.instructions.append(IRInstruction(IRType.LABEL, start_label))
        
        # Condition check
        if node.condition:
            condition = self.visit(node.condition)
            self.instructions.append(IRInstruction(IRType.JUMP_IF_FALSE, end_label, condition))
        
        # Body
        self.visit(node.body)
        
        # Update
        self.instructions.append(IRInstruction(IRType.LABEL, update_label))
        if node.update:
            self.visit(node.update)
        
        # Jump back to condition
        self.instructions.append(IRInstruction(IRType.JUMP, start_label))
        
        # Loop end
        self.instructions.append(IRInstruction(IRType.LABEL, end_label))
    
    def visit_PrintStmt(self, node):
        """Visit a print statement."""
        value = self.visit(node.expr)
        self.instructions.append(IRInstruction(IRType.PRINT, value))
    
    def visit_BinaryExpr(self, node):
        """Visit a binary expression."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        result = self.new_temp()
        
        if node.operator == TokenType.PLUS:
            self.instructions.append(IRInstruction(IRType.ADD, result, left, right))
        elif node.operator == TokenType.MINUS:
            self.instructions.append(IRInstruction(IRType.SUB, result, left, right))
        elif node.operator == TokenType.MULTIPLY:
            self.instructions.append(IRInstruction(IRType.MUL, result, left, right))
        elif node.operator == TokenType.DIVIDE:
            self.instructions.append(IRInstruction(IRType.DIV, result, left, right))
        elif node.operator == TokenType.EQUAL:
            self.instructions.append(IRInstruction(IRType.EQ, result, left, right))
        elif node.operator == TokenType.NOT_EQUAL:
            self.instructions.append(IRInstruction(IRType.NE, result, left, right))
        elif node.operator == TokenType.LESS:
            self.instructions.append(IRInstruction(IRType.LT, result, left, right))
        elif node.operator == TokenType.GREATER:
            self.instructions.append(IRInstruction(IRType.GT, result, left, right))
        elif node.operator == TokenType.LESS_EQUAL:
            self.instructions.append(IRInstruction(IRType.LE, result, left, right))
        elif node.operator == TokenType.GREATER_EQUAL:
            self.instructions.append(IRInstruction(IRType.GE, result, left, right))
        
        return result
    
    def visit_UnaryExpr(self, node):
        """Visit a unary expression."""
        expr = self.visit(node.expr)
        result = self.new_temp()
        
        if node.operator == TokenType.MINUS:
            self.instructions.append(IRInstruction(IRType.SUB, result, "0", expr))
        
        return result
    
    def visit_FunctionCall(self, node):
        """Visit a function call."""
        # Evaluate and push arguments
        args = []
        for arg in node.args:
            arg_val = self.visit(arg)
            self.instructions.append(IRInstruction(IRType.PARAM, arg_val))
            args.append(arg_val)
        
        # Call function
        result = self.new_temp()
        args_str = ", ".join(args)
        self.instructions.append(IRInstruction(IRType.CALL, result, node.name, args_str))
        
        return result
    
    def visit_IntLiteral(self, node):
        """Visit an integer literal."""
        return str(node.value)
    
    def visit_FloatLiteral(self, node):
        """Visit a float literal."""
        return str(node.value)
    
    def visit_StringLiteral(self, node):
        """Visit a string literal."""
        return f'"{node.value}"'
    
    def visit_Identifier(self, node):
        """Visit an identifier."""
        return node.name

class CodeGenerator:
    def __init__(self, ir_instructions):
        self.ir = ir_instructions
        self.output = []
        self.indent_level = 0
        self.labels = {}
    
    def generate(self):
        """Generate target code from IR instructions."""
        # First pass: record all label positions
        for i, instr in enumerate(self.ir):
            if instr.op_type == IRType.LABEL:
                self.labels[instr.result] = i
        
        # Add Python runtime header
        self.output.append("# Generated Python code")
        self.output.append("# Runtime")
        self.output.append("def _runtime_print(value):")
        self.output.append("    print(value)")
        self.output.append("")
        
        # Second pass: generate code
        i = 0
        while i < len(self.ir):
            instr = self.ir[i]
            i += 1
            
            if instr.op_type == IRType.LABEL:
                if instr.result.startswith("function_"):
                    # Function definition
                    func_name = instr.result[9:] 
                    self.output.append(f"def {func_name}():")
                    self.indent_level += 1
                else:

                    self.output.append(f"# {instr.result}")
            elif instr.op_type == IRType.DECLARE:
                if instr.arg1 == "int":
                    self.add_line(f"{instr.result} = 0")
                elif instr.arg1 == "float":
                    self.add_line(f"{instr.result} = 0.0")
                elif instr.arg1 == "string":
                    self.add_line(f"{instr.result} = \"\"")
                else:
                    self.add_line(f"{instr.result} = None")
            elif instr.op_type == IRType.ASSIGN:
                self.add_line(f"{instr.result} = {instr.arg1}")
            elif instr.op_type == IRType.ADD:
                self.add_line(f"{instr.result} = {instr.arg1} + {instr.arg2}")
            elif instr.op_type == IRType.SUB:
                self.add_line(f"{instr.result} = {instr.arg1} - {instr.arg2}")
            elif instr.op_type == IRType.MUL:
                self.add_line(f"{instr.result} = {instr.arg1} * {instr.arg2}")
            elif instr.op_type == IRType.DIV:
                self.add_line(f"{instr.result} = {instr.arg1} / {instr.arg2}")
            elif instr.op_type == IRType.JUMP:
                target_idx = self.labels.get(instr.result)
                if target_idx is not None:
                    # Find next instruction after the jump target
                    self.add_line(f"# Jump to {instr.result}")
                    self.add_line(f"goto {instr.result}")
            elif instr.op_type == IRType.JUMP_IF_TRUE:
                self.add_line(f"if {instr.arg1}:")
                self.indent_level += 1
                self.add_line(f"goto {instr.result}")
                self.indent_level -= 1
            elif instr.op_type == IRType.JUMP_IF_FALSE:
                self.add_line(f"if not {instr.arg1}:")
                self.indent_level += 1
                self.add_line(f"goto {instr.result}")
                self.indent_level -= 1
            elif instr.op_type == IRType.CALL:
                args = instr.arg2 if instr.arg2 else ""
                self.add_line(f"{instr.result} = {instr.arg1}({args})")
            elif instr.op_type == IRType.RETURN:
                if instr.result:
                    self.add_line(f"return {instr.result}")
                else:
                    self.add_line("return")
                # Decrease indent if we're in a function
                if self.indent_level > 0:
                    self.indent_level -= 1
            elif instr.op_type == IRType.PARAM:
                pass
            elif instr.op_type == IRType.PRINT:
                self.add_line(f"_runtime_print({instr.result})")
            elif instr.op_type == IRType.EQ:
                self.add_line(f"{instr.result} = {instr.arg1} == {instr.arg2}")
            elif instr.op_type == IRType.NE:
                self.add_line(f"{instr.result} = {instr.arg1} != {instr.arg2}")
            elif instr.op_type == IRType.LT:
                self.add_line(f"{instr.result} = {instr.arg1} < {instr.arg2}")
            elif instr.op_type == IRType.GT:
                self.add_line(f"{instr.result} = {instr.arg1} > {instr.arg2}")
            elif instr.op_type == IRType.LE:
                self.add_line(f"{instr.result} = {instr.arg1} <= {instr.arg2}")
            elif instr.op_type == IRType.GE:
                self.add_line(f"{instr.result} = {instr.arg1} >= {instr.arg2}")
        

        self.output.append("\n# Execute main function if it exists")
        self.output.append("if 'main' in globals():")
        self.output.append("    main()")
        
        return "\n".join(self.output)
    
    def add_line(self, line):
        """Add a line of code with proper indentation."""
        self.output.append("    " * self.indent_level + line)


def compile_code(source_code, target_lang="python"):
    """Compile source code to target language."""
    # 1. Lexical Analysis
    print("1. Performing lexical analysis...")
    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()
    except SyntaxError as e:
        print(f"Lexical error: {e}")
        return None
    
    # 2. Syntax Analysis
    print("2. Performing syntax analysis...")
    parser = Parser(tokens)
    try:
        ast = parser.parse()
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        return None
    
    # 3. Semantic Analysis
    print("3. Performing semantic analysis...")
    analyzer = SemanticAnalyzer()
    success, errors = analyzer.analyze(ast)
    if not success:
        print("Semantic errors:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    # 4. IR Generation
    print("4. Generating intermediate representation...")
    ir_gen = IRGenerator()
    ir_instructions = ir_gen.generate(ast)
    
    # 5. Code Generation
    print("5. Generating target code...")
    code_gen = CodeGenerator(ir_instructions)
    target_code = code_gen.generate()
    
    return target_code

def main():
    print("Enter your source code below (type 'END' on a new line to finish):")
    source_code_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        source_code_lines.append(line)
    
    source_code = "\n".join(source_code_lines)
    
    target_code = compile_code(source_code)
    if target_code:
        print("Compilation successful. Generated code:")
        print("----------------------------------------")
        print(target_code)

if __name__ == "__main__":
    main()