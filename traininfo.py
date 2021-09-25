from torch.utils.tensorboard import SummaryWriter, writer
import shutil

STOREDIR="runs/lstm"
class TrainInfo(object):
    def __init__(self):
        shutil.rmtree(STOREDIR,ignore_errors=True)
        self.writer = SummaryWriter(STOREDIR)

    def add_scalar(self,name, loss, epoch):
        self.writer.add_scalar(name, loss, epoch)
    
    def add_graph(self,model, data):
        self.writer.add_graph(model, data)
