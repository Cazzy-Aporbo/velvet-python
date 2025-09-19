import random
import pprint

def kelvin_to_rankine(kelvin):
    rankine = kelvin * 1.8
    return rankine

def convert_temperatures_function(temperatures):
    rankine_temperatures = []
    for temp in temperatures:
        rankine = kelvin_to_rankine(temp)
        rankine_temperatures.append(rankine)
    return rankine_temperatures

def convert_temperatures_lambda(temperatures):
    rankine_temperatures = list(map(lambda temp: temp * 1.8, temperatures))
    return rankine_temperatures

kelvin_temperatures = [random.uniform(200, 400) for _ in range(5)]
rankine_temperatures_function = convert_temperatures_function(kelvin_temperatures)
rankine_temperatures_lambda = convert_temperatures_lambda(kelvin_temperatures)


pp = pprint.PrettyPrinter(indent=4)

print("Kelvin Temperatures:")
pp.pprint(kelvin_temperatures)

print("Rankine Temperatures (Using Function):")
pp.pprint(rankine_temperatures_function)

print("Rankine Temperatures (Using Lambda Function):")
pp.pprint(rankine_temperatures_lambda)
