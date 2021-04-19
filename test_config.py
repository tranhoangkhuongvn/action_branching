from config import *


if __name__ == '__main__':
	test_config = Configuration('./config.json')
	print(test_config.env_name)
