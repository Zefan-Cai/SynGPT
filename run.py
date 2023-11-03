

from utils.arguments import get_arguments
from utils.utils import set_seed

from training.trainer_DeepDDs_divide import Trainer as Trainer_DeepDDs_divide
from training.trainer_DeepDDs import Trainer as Trainer_DeepDDs

from training.trainer_GraphSynergy import Trainer as Trainer_GraphSynergy
from training.trainer_GraphSynergy_divide import Trainer as Trainer_GraphSynergy_divide

from training.trainer_DeepSynergy import Trainer as Trainer_DeepSynergy
from training.trainer_DeepSynergy_divide import Trainer as Trainer_DeepSynergy_divide

from training.trainer_PRODeepSyn import Trainer as Trainer_PRODeepSyn
from training.trainer_PRODeepSyn_divide import Trainer as Trainer_PRODeepSyn_divide

from training.trainer_AuDNNsynergy import Trainer as Trainer_AuDNNsynergy
from training.trainer_AuDNNsynergy_divide import Trainer as Trainer_AuDNNsynergy_divide


if __name__ == "__main__":

    args = get_arguments()
    set_seed(args)

    for data_partition_idx in range(args.start_fold, args.end_fold):

        args.data_partition_idx = data_partition_idx

        if args.model == "DeepDDs":
            if args.noisylabel == True:
                trainer = Trainer_DeepDDs_divide(args)
                trainer.train()
            else:
                trainer = Trainer_DeepDDs(args)
                trainer.train()
        elif args.model == "GraphSynergy":
            if args.noisylabel == True:
                trainer = Trainer_GraphSynergy_divide(args)
                trainer.train()
            else:
                trainer = Trainer_GraphSynergy(args)
                trainer.train()
        elif args.model == "DeepSynergy":
            if args.noisylabel == True:
                trainer = Trainer_DeepSynergy_divide(args)
                trainer.train()
            else:
                trainer = Trainer_DeepSynergy(args)
                trainer.train()
        elif args.model == "PRODeepSyn":
            if args.noisylabel == True:
                trainer = Trainer_PRODeepSyn_divide(args)
                trainer.train()
            else:
                trainer = Trainer_PRODeepSyn(args)
                trainer.train()
        elif args.model == "AuDNNsynergy":
            if args.noisylabel == True:
                trainer = Trainer_AuDNNsynergy_divide(args)
                trainer.train()
            else:
                trainer = Trainer_AuDNNsynergy(args)
                trainer.train()
        


