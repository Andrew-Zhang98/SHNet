import os.path as osp
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args, epoch_i):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    method_names = args.methods.split('+')

    threads = []
    test_paths = args.test_paths.split('+')

    all_res={}
    for dataset_setname in test_paths:
        for method in method_names:

            pred_dir_all = osp.join(pred_dir, dataset_setname, method)

            gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/GT'

            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method, dataset_setname, output_dir, cuda=True, epoch_i = epoch_i)
            threads.append(thread)
    for thread in threads:
        res = thread.fast_run()
        all_res[thread.dataset] =res
        # import pdb; pdb.set_trace()
        # print(thread.run())
        # print(thread.fast_run())
    return all_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_test_path_root', type=int, default=256, help='testing size')
    parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
    parser.add_argument('--test_path',type=str,default='/4T/wenhu/dataset/RGB-SOD/', help='test dataset path')
    opt = parser.parse_args()
    evaluate()