class CyclicLR(callbacks.Callback):
    def __init__(self, base_lr=0.001,
                 max_lr=0.006, 
                 step_size=2000,
                 policy='triangular',
                 scale_fn=None,
                 mode=None,
                 gamma=0.99994):
        
        super(CyclicLR, self).__init__()
        
        self.mode = mode
        self.step_size = step_size
        self.clr_iterations = 0
        self.history = {}
        self.delta = max_lr - base_lr
        self.base_lr = base_lr

        if scale_fn is None:
            if policy is 'triangular':
                self.scale_fn = lambda x: 1
                self.mode = 'cycle'
            elif policy is 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.mode = 'cycle'
            elif policy is 'exp_range':
                self.gamma = gamma
                self.scale_fn = lambda x: self.gamma**(x)
                self.mode = 'iterations'
        else:
          self.scale_fn = scale_fn

        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        
        if self.mode == 'iterations':
            clr = self.base_lr + self.delta*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        else:
            clr = self.base_lr + self.delta*np.maximum(0, (1-x))*self.scale_fn(cycle)
        return clr 
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.lr_lst = []

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.learning_rate, self.base_lr)
            
    def on_batch_end(self, epoch, logs=None):
        self.clr_iterations += 1
        lr_value = K.get_value(self.model.optimizer.learning_rate)
        logs['lr'] = lr_value
        self.lr_lst.append(lr_value)
        K.set_value(self.model.optimizer.learning_rate, self.clr())
        
    def on_train_end(self, logs=None):
        self.history['lr'] = self.lr_lst
