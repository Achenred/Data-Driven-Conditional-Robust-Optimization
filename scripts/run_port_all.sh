for port in `seq 0 50`; do #`seq 0 50`  #5 6 40 10 49 50 22 23 26 15 #sequence of numbers or randomly generated 10 numbers to pick the portfolio combination to optimize
python path/data/generate_port_data.py $port

for y in  2013 ; do #`seq 2012 2014`
python path/data/get_data.py $y

for lr in 0.01; do # 0.01 0.001 0.0001

python code/train_nn/main_mark.py portfolio port_gen scripts/log/ path/data/train_$y.csv path/data/test_$y.csv --objective one-class --lr $lr --n_epochs 800 --batch_size 1100 --weight_decay 0.01 --pretrain False --normal_class 0 
python code/solver/start-port_mark_sigma.py $y $lr -1000 $port

python code/solver/start-port_elli.py $y $port

for alpha in 0.5 ; do #0.0001 0.01 0.3 0.6 0.9 0.99 0.999    0 0.01 0.3 0.6 0.9 0.99 0.999 1   
for net in 1; do #1 2 3
for classes in 2; do #`seq 2 5`
for b in 0.15 ; do #0.0004; do #0.0001 0.0002 0.00015 0.0003 0.00009  0.105 0.11
for eps in 0.5; do #0.01 0.3 0.6 0.9 0.99 

echo $y
echo $lr
echo $alpha
echo $net
echo $classes
echo $b
echo $eps


python code/train_nn/main_AE.py portfolio port_soft_assign_AE${net} scripts/log/portfolio path/data/train_$y.csv path/data/test_$y.csv --n_clusters $classes --objective one-class --year $y --beta $b --alpha $alpha --eps $eps --lr $lr --n_epochs 800 --batch_size 1100 --weight_decay 0.01 --pretrain False --normal_class 0 
python code/solver/start-port-convexhull.py $alpha $lr $classes $y 1 $b $eps $net -1000 $port  #w/o alpha 


python code/train_nn/main_deep_kmeans.py portfolio deep_kmeans_AE scripts/log/portfolio path/data/train_$y.csv path/data/test_$y.csv --n_clusters $classes --objective one-class --year $y --beta $b --alpha $alpha --lr $lr --n_epochs 800 --batch_size 2000 --weight_decay 0.01 --pretrain False --normal_class 0 
python code/solver/start-port_kmeans_mark_sigma.py $alpha $lr $classes $y 1 $b $eps $net $port


done
done
done
done
done

done
done
done

#increase beta - 
#decrease beta - 