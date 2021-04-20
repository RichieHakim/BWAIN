% program: licking habituation

for ii = 1:190
    giveReward2(1, 1, 150, 200, 1 , 1 , 0.1);
    ii
    interval = randi([8 32])
    pause(interval)
end