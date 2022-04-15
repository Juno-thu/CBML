from tap import Tap


class GlobalParam(Tap):
    dataset: str = 'demo'  # dataset name: JD_click_data_demo
    batchSize: int = 100  # input batch size
    hiddenSize: int = 100  # hidden state size
    epoch: int = 30  # the number of epochs to train for
    lr: float = 0.001  # learning rate [0.001, 0.0005, 0.0001]
    lr_dc: float = 0.1  # learning rate decay rate
    lr_dc_step: int = 3  # the number of steps after which the learning rate decay
    l2: float = 1e-05  # l2 penalty [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    step: int = 1  # gnn propagation steps
    patience: int = 10  # the number of epoch to wait before early stop
    nonhybrid: bool = False  # only use the global preference to predict
    validation: bool = False  # validation
    valid_portion: float = 0.1  # split the portion of training set as validation set
    cluster_num: int = 16  # cluster number for soft clustering
    memory_dim: int = 0  # memory dim


opt: GlobalParam = GlobalParam().parse_known_args()[0]
