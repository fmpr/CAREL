from AAPI import *

def log(*args, **kwargs):
	"""Special function for printing to the Aimsun log with similar behaviour to the print() function"""
	if 'level' not in kwargs:
		kwargs['level'] = 1
	prefix = '[DeepRL][L%d]' % (kwargs['level'],)
	AKIPrintString(prefix + ' '.join([str(x) for x in args]))
