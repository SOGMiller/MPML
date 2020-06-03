import matplotlib.pyplot as plt
import uuid
import math

unique_filename = str(uuid.uuid4()) 

import pickle

feature_impacts = []

def plotGraph(feature_impact_list,title,folder_name):

	feature_impacts.append(feature_impact_list)

	title_list = []

	fileName = title+".png"

	x = get_features(feature_impact_list)
	y = get_impact_scores(feature_impact_list)

	labels = x

	plt.plot(x,y, '--bo')

	plt.xticks(x, labels, rotation='vertical')
	plt.xlabel("Features")
	plt.ylabel("Impact Score")
	plt.title(title)
	plt.tight_layout()
	plt.grid()
	plt.rcParams["figure.figsize"] = (15,15)

	plt.savefig(folder_name+"/"+fileName)
	# plt.savefig("saves/"+fileName)

	plt.close()

def quad_plot(title_list,algo_name):

	graphNum = len(title_list)
	row1 = math.floor(graphNum/2)  


	fig, axs = plt.subplots(2,2)

	axis = axs.flat
	count = 0

	for impact_lits in feature_impacts:
		x = get_features(impact_lits)
		y = get_impact_scores(impact_lits)

		axis[count].plot(x,y,'--bo')
		axis[count].set_title("Instance #{} with {}".format(title_list[count],algo_name))
		axis[count].grid()
		count+=1

	for ax in axs.flat:
		ax.set(xlabel='Features', ylabel='Impact Score')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
	# for ax in axs.flat:
		# ax.label_outer()

	fig.set_size_inches(15, 15)
	fig.autofmt_xdate()
    
	fig.savefig("saves/Quad.png",bbox_inches='tight',dpi=100)

	# fig.close()

def get_impact_scores(feature_impact_list):
	impact_scores = []

	for x in feature_impact_list:
		impact_scores.append(x[1])

	return impact_scores

def get_features(feature_impact_list):
	impact_scores = []
	for x in feature_impact_list:
		impact_scores.append(x[0])
	return impact_scores

def save_models(model,filename):
	with open("saves/"+filename, 'wb') as save_file:
		pickle.dump(model,save_file)