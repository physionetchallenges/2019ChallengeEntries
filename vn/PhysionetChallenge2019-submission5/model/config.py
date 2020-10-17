import tensorflow as tf

flags = tf.app.flags.FLAGS

class Config:
	def __init__(self):
		self.user_flags = []
		self.get_config()
		#self.print_user_flags()
	

	def get_config(self):
		DEFINE_string = self.DEFINE_string
		DEFINE_integer = self.DEFINE_integer
		DEFINE_float = self.DEFINE_float
		DEFINE_boolean = self.DEFINE_boolean
		DEFINE_boolean('istest', True, '')

		# Data setting
		DEFINE_string('training_set', 'A1/A2/A3/B1/B2/B3', 'sep:/')
		DEFINE_string('padding', 'B', 'F:forward padding / B:backward padding')

		## Training setting
		DEFINE_integer('batch_size', 1, '')

		## Model setting
		DEFINE_integer('input_length', 50, '')
		DEFINE_integer('model_depth', '3', 'model depth:2/3/4/5/6/7/8')
		DEFINE_integer('output_size', 2, '4 in case of uncertainty')
		DEFINE_integer('num_sample', 1, 'the number of MonteCarlo sampling')
		DEFINE_string('name', 'NN1', 'name of model')

		## Loading/Saving
		DEFINE_string('DIR_SOURCE', './model/weight/', '') #
		DEFINE_string('TRAIN_SCOPE', flags.name, '')
		#DEFINE_string('DIR_SOURCE', flags.DIR_SAVE, '')
		DEFINE_string('LOAD_SCOPE', 'NN1', '')
		DEFINE_string('FILE_HISTORY', 'history.csv', '')



	def DEFINE_string(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_string(name, default_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_integer(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_integer(name, default_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_float(self, name, defualt_value, doc_string):
		tf.app.flags.DEFINE_float(name, defualt_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_boolean(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
		self.user_flags.append(name)

	def print_user_flags(self, line_limit = 80):
		temp = []
		for flag_name in sorted(self.user_flags):
			value = "{}".format(getattr(flags, flag_name))
			log_string = flag_name
			log_string += "." * (line_limit - len(flag_name) - len(value))
			log_string += value
			temp.append(log_string)
			print(log_string)
		return temp
