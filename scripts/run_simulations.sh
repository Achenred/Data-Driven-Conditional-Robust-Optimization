for l in 10; do #for l in `seq 1 2`; do
for t in 10; do #for t in 1 2; do
for s in 10; do #for t in 1 2; do

:'
python code/generator/gen2.py $t 10475 $s $l 200 | tr -d '\[' | tr -d '\]' > temp.txt
touch scripts/selectiondata16/train-$t-$l-$s.txt
touch scripts/selectiondata16/test-$t-$l-$s.txt
cat temp.txt | sed -n "1,500p" > scripts/selectiondata16/train-$t-$l-$s.txt
cat temp.txt | sed -n "501,10500p" > scripts/selectiondata16/test-$t-$l-$s.txt
'
python code/solver/start-sim_elli.py $t $l $s

for net in 3; do
for rep in 1; do #for rep in `seq 1 3`; do

python code/train_nn/main_mark.py generated mine_gen scripts/log/ scripts/selectiondata16/train-$t-$l-$s.txt scripts/selectiondata16/test-$t-$l-$s.txt --objective one-class --lr 0.001 --n_epochs 800 --batch_size 3000 --weight_decay 0.001 --pretrain False --normal_class 1 > temp-${rep}.txt
python code/solver/start-sim_mark_sigma.py $t $l $s


python code/train_nn/main_AE.py mine soft_assign_AE1 scripts/log/portfolio scripts/selectiondata16/train-$t-$l-$s.txt scripts/selectiondata16/test-$t-$l-$s.txt --n_clusters 2 --objective one-class --year 0 --beta 1 --alpha 0.5 --eps 0.5 --lr 0.01 --n_epochs 500 --batch_size 30000 --weight_decay 0.01 --pretrain False --normal_class 0 
python code/solver/start-sim.py 0.5 0 2 $t $l $s 0  #alpha, val, classes 

done

done

done
done
done

