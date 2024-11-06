#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-IR-LIDAR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_RGBIRLIDARINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save trial_RGBINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method IR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_IRINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method LIDAR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_LIDARINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-IR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_RGBIRINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-LIDAR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save trial_RGBLIDARINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method IR-LIDAR-INT --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_IRLIDARINT
#
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-IR-LIDAR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_RGBIRLIDAR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save trial_RGB
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method IR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_IR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method LIDAR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_LIDAR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-IR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_RGBIR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method RGB-LIDAR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save trial_RGBLIDAR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd_tiny.json --method IR-LIDAR --loss 1.0*L2 --epochs 2 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4 --save trial_IRLIDAR




#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd.json --method RGB-LIDAR --loss 1.0*L2 --epochs 10 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save MMDCE_KITTIMMD_e10_b8_L2_RGBLIDAR
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd.json --method RGB --loss 1.0*L2 --epochs 10 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 8 --save MMDCE_KITTIMMD_e10_b8_L2_RGB
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd.json --method RGB-LIDAR-INT --loss 1.0*L2 --epochs 10 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 6 --save MMDCE_KITTIMMD_e10_b8_L2_RGBLIDARINT
#python main.py --dataset KITTIMMD --list_data ../list_data/kitti_mmd.json --method RGB-INT --loss 1.0*L2 --epochs 10 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 6 --save MMDCE_KITTIMMD_e10_b8_L2_RGBINT

#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method IR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-IR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method IR-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-IR-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method IR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-IR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method IR-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day_tiny.json --method RGB-IR-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4

#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method IR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-IR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method IR-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-IR-LIDAR --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method IR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-IR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method IR-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4
#python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/night --list_data ../list_data/mmdce_night_tiny.json --method RGB-IR-LIDAR-INT --loss 1.0*L2 --epochs 1 --decay 5,8,10 --gamma 1.0,0.2,0.04 --batch_size 4



python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGB
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method IR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_IR
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method LIDAR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_LIDAR
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-IR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBIR
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-LIDAR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBLIDAR
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method IR-LIDAR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_IRLIDAR
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-IR-LIDAR --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBIRLIDAR

python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method IR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_IRINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method LIDAR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_LIDARINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-IR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBIRINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-LIDAR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBLIDARINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method IR-LIDAR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_IRLIDARINT
python main.py --dataset MMDCE --path_mmdce /home/viplab/Dataset/MMDCE/day --list_data ../list_data/mmdce_day.json --method RGB-IR-LIDAR-INT --loss 1.0*L2 --epochs 20 --decay 8,12,16,20 --gamma 1.0,0.2,0.04,0.08 --batch_size 16 --gpus 0,1,2,3 --save MMDCE_MMDCEDAY_e20_b16_L2_RGBIRLIDARINT
