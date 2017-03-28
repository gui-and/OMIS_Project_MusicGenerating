import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
import matplotlib.patches as mpatches

def plot_loss(filename):
	loss = {}

	#Read values stored in dict in file.
	with open(filename, 'r') as csvfile:
	    reader = csv.DictReader(csvfile)
	    for row in reader:
	       #using ast.literal_eval() because data is in a str
	       loss['discriminator'] = ast.literal_eval(row['discriminator'])
	       loss['generator'] = ast.literal_eval(row['generator'])



	epoch = list(range(1, len(loss['discriminator']) +1))
	fig = plt.figure(1)

	plt.title("Loss observation during %d epoch" % len(loss['discriminator']))

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(epoch, loss['discriminator'], color="red")
	plt.plot(epoch, loss['generator'], color="orange")

	red_patch = mpatches.Patch(color='red', label='Discriminator')
	orange_patch = mpatches.Patch(color='orange', label='Generator')
	plt.legend(handles=[red_patch, orange_patch])

	plt.show()

	plt.savefig('loss.png', bbox_inches='tight')
