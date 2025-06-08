from cgenerator import CaseGenerator
from model import TransformerModel
from loadmod import TransformerLoader
from datapipe import Datapipe
from utils import CommonUtil
import os
from sql import SQLAdministrator



if __name__ == '__main__':
    program_name = 'Gen'
    adm = SQLAdministrator(program_name)
    adm.init_sql()
    adm.save_root_data(os.path.join('root_data', 'root_mod.xlsx'),
                       os.path.join('root_data', 'vocab.xlsx'))
    cg = CaseGenerator(generator_name= program_name,
                       root_elems_list= adm.root.get_rule_element_list(),
                       contradictory= adm.root.get_contradictory(),
                       noise_pool = adm.root.get_noises(),
                       synonym=adm.root.get_synonym(),
                       modularized=adm.root.get_modularized(),
                       accessory=adm.root.get_accessory())
    cg.generate_train_val_cases(N_times=1, begin_num=0, train_val_ratio=10)

# -----------------------------------------------------------------
    train_dp_iter, val_dp_iter, vocabs = \
        Datapipe.make_standard_train_val_datapipe(cg.std_train_paths, cg.std_val_paths,
                                                  train_bs=100, val_bs=50)
    model = TransformerModel(vocabs, save_dir_path='ModelDict')
    model.init_loss_fn_and_opt()
    model.print_config()
    model.train_model(train_dp_iter, val_dp_iter, total_epochs=100, epoch_begin=0, del_used_dataset=True,
                      print_interval=2000, print_instances=True)

# -----------------------------------------------------------------
#
#     path_checkpoint = os.path.join('TransformerModelDict', 'checkpoint_latest.pth')
#     ml = TransformerLoader(path_checkpoint)
#     train_dp_iter, val_dp_iter, vocabs = \
#         Datapipe.make_standard_train_val_datapipe(cg.std_train_paths, cg.std_val_paths, ml.vocabs,
#                                                   train_bs=100, val_bs=50, add_cls=True)
#
#     ml.train_model(train_dp_iter, val_dp_iter, del_used_dataset=True, total_epochs=20,
#                     print_interval=2000, print_instances=True)

# -----------------------------------------------------------------
#     CommonUtil.transform_test_data('./root_data/test4.txt', './root_data/test_trans.txt')
#     path_checkpoint = os.path.join('TransformerModelDict', 'checkpoint_best.pth')
#     ml = TransformerLoader(path_checkpoint)
#     dp_iter, vocabs = Datapipe.make_standard_datapipe('./root_data/test_trans.txt', ml.vocabs, batch_size=1, batch_num=1)
#     ml.test_model(dp_iter, decode_method='top_k_greedy_decode', top_k=1)
    #
    # ml.test_model(dp_iter, decode_method='greedy_decode', top_k=1)
    #

    # test_path = cg.generate_test_cases()
    # test_dp = make_test_dp(test_path, ml.vocabs)
    # ml = ModelLoader(path_model, path_checkpiont)
    # result = ml.use_model(['发热','往来寒热','脉弦'])
    # print(result)


