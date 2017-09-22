import ConfigParser
import ast

my_list = ast.literal_eval(config.get("section", "option"))
print(type(my_list))
print(my_list)
