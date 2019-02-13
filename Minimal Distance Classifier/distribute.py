def distribute(x,y):
    orphan=[]
    # initial list
    for i in range(x):
        element =[i+1,0]
        orphan.append(element)
    #if there is money remaining, keep looping
    round1 = 1;
    while(y>0):

        for i in range(x):
            temp= i +round1
            if y>=temp:
                orphan[i][1]=orphan[i][1]+temp;

                y=y-temp
            else:
                orphan[i][1]=orphan[i][1]+y

                y=y-temp
                break
        round1 = round1 + 1
    for i in orphan:
        print("orphan index %i, money he/she got $%i"%(i[0],i[1]))


x=int(input("please input the number of orphans : "))
y=int(input("please input the number of money :"))
distribute(x,y)
