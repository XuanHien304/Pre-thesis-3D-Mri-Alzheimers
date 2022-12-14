from src.trainer import BaselineClassifier
import torch
import json
import time
import os
from pprint import pprint


def train():
	# Initialisation
	best_score = 0	
	best_val_loss = float('inf') # Initial best validation loss

	# Load parameters
	params = json.load(open('./config/config_train.json', 'r'))
	pprint(params)

	# LOAD TRAINER
	trainer = BaselineClassifier(**params)
	# Set up DataLoader
	train_iterator, val_iterator = trainer.set_up_training_data()
	# Set up training params
	model, optimizer, loss_criteria, lr_scheduler, device = trainer.set_up_training()
	
	# Start measuring training time
	start_time = time.monotonic()

	# TRAINING
	for epoch in trainer.MAX_EPOCHS:

		# Train, calculate validation loss and score
		train_loss = trainer.epoch_train(model, train_iterator, optimizer, loss_criteria, device, mb)
		val_loss, new_score = trainer.epoch_evaluate(model, val_iterator, loss_criteria, device, mb)

		# Update learning rate (according to validation f1 score)
		lr_scheduler.step(new_score)

		# Save model
		if not os.path.exists('./models'):
			os.makedirs('./models')

		if best_score < new_score:
			best_score = new_score
			nonimproved_epoch = 0
			torch.save({"model": model.state_dict(),
						"optimizer": optimizer.state_dict(),
						"best_score": best_score,
						"epoch": epoch,
						"lr_scheduler": lr_scheduler.state_dict()}, 'models/retina_epoch{}_score{:.4f}.pth'.format(epoch, new_score)) # Make sure the folder `models/` exists
		else:
			nonimproved_epoch += 1
		
		if nonimproved_epoch > 10:
			print('Early stopping. Model not improving.')
			break
		
		if time.time() - start_time > trainer.TRAINING_TIME_OUT:
			print('Early stopping. Out of time.')

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model.state_dict(), trainer.SAVE_MODEL_PATH)

	end_time = time.monotonic()
	epoch_mins, epoch_secs = trainer.convert_time(start_time, end_time)
	print(f'Training time: {epoch_mins}m {epoch_secs}s')
	print('TRAINING DONE')

if __name__ == '__main__':
	train()