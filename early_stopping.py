import torch
class EarlyStopping:
	"""docstring for EarlyStopping"""
	def __init__(self, patience, eps= 0, save_path= 'checkpoint/checkpoint.pt'):
		super().__init__()
		self.patience, self.eps, self.save_path= patience, eps, save_path
		self.best_score, self.counter, self.flag= None, 0, False
	
	def __call__(self, val_loss, model):

		score= -val_loss
		if self.best_score is None:
			self.best_score= score
			self.save_checkpoint(model)
		elif score< self.best_score- self.eps:
			self.counter+= 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter>= self.patience:
				self.flag= True
		else:
			self.best_score= score
			self.save_checkpoint(model)
			self.counter= 0

	def save_checkpoint(self, model):
		torch.save(model.state_dict(), self.save_path)