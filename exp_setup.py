import numpy as np
import numpy.random as rnd

def get_random_q(num_species=4):
    rng = rnd.default_rng()
    q_1 = np.concatenate([rng.normal(10,1, size=1),rng.normal(5, 1, size=1),rng.normal(7, 1, size=1),rng.normal(15, 2, size=1)])#speed
    q_2 = np.concatenate([rng.normal(90,5, size=1),rng.normal(150, 9, size=1),rng.normal(120, 10, size=1),rng.normal(75, 12, size=1)]) #footprint
    q_3 = np.concatenate([rng.normal(8,2, size=1),rng.normal(5, 1, size=1),rng.normal(16, 1, size=1),rng.normal(7, 2, size=1)]) #payload
    q_4 = np.concatenate([rng.normal(2,1, size=1),rng.normal(3, 1, size=1),rng.normal(2, 1, size=1),rng.normal(3, 2, size=1)]) #reach
    q_5 = rng.normal(np.random.randint(25,30), 1, size=num_species) #weight
    q_6 = rng.normal(np.random.randint(210,230), 15.4, size=num_species) #sensing frequency
    q_7 = rng.normal(np.random.randint(58,60), 0.8, size=num_species) #sensing range
    q_8 = rnd.choice([0,1,2,3,4], num_species) #color
    q_9 = rnd.choice(range(10,20), num_species) #battery
    Q = np.array([q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9]).T
    return Q

def get_deterministic_q():
    Q_master = np.array([   [  9.23918352,  11.90620543,  10.53292464,   9.59601526, 8.09180044,   5.3907205 ,   4.36013104,   6.26982438, 5.62048407,   4.33589295,   6.83473728,   6.93909328, 5.466237  ,   4.89535722,   7.37487731,  13.91198224, 14.8902714 ,  20.1714527 ,  13.42318422,  15.25451451],   
                            [ 83.49960611,  91.42168321,  87.46871472,  86.05393088,85.46500857, 152.72279577, 142.04504698, 157.27231004,142.53179087, 145.87187006, 134.45525572, 125.61530258,116.89311174, 125.06972235, 120.37085389,  74.56449124,58.97478981,  91.61323028,  80.57563977,  68.30003836],
                            [ 10.89758117,   5.93457914,   6.70158357,   9.96495997,11.35077763,   5.91330245,   4.12606487,   4.61532922,4.57792751,   4.03014111,  17.25782298,  14.81293339,15.01662868,  14.88700865,  16.3928066 ,   6.71303366, 7.02389812,   6.27550929,   2.09331897,   8.93246973],
                            [  3.35040671,   1.14991181,   2.03221372,   0.57500585,2.01644894,   3.41145602,   2.65880892,   2.49255253,4.52075551,   1.96576587,   2.3917984 ,   1.78786766,2.27236423,   2.60576371,   3.37611105,   3.60091245,2.29526818,   3.26773863,   3.66587264,   4.04958294],
                            [ 27.35963654,  28.70833559,  28.17720895,  28.35548257,28.52188259,  25.92882973,  26.39641364,  28.84505038,    28.0369515 ,  26.23950531,  26.4445487 ,  28.43585418,    27.06816073,  26.35838129,  28.74999209,  27.91573835,27.29551519,  27.39695175,  27.43213408,  27.90922054],
                            [217.37665023, 254.46721127, 214.58548929, 212.30796155,207.6643806 , 232.82418229, 219.35875565, 225.80245733,192.92778782, 211.88948643, 222.73195506, 224.40770549,193.56940232, 209.5525548 , 214.41225727, 207.96557007,196.76240068, 215.34573519, 246.79628606, 232.69723099],
                            [ 58.77190383,  60.3231273 ,  60.67383826,  58.73472194,    57.12200567,  58.5370682 ,  59.53347581,  59.50739216,    58.68445876,  58.97691257,  59.13420197,  58.46502442,    59.27933208,  59.01305319,  58.68714375,  59.54004468,  58.97430914,  57.47285604,  59.31326026,  57.56596041],
                            [  1.  ,   1.  ,   0.  ,   2.  ,4.  ,   3.  ,   4.  ,   0.  ,1.  ,   1.  ,   0.  ,   0.  ,0.  ,   4.  ,   1.  ,   0.  ,2.  ,   1.  ,   3.  ,   4.  ],
                            [ 12.  ,  17.  ,  12.  ,  14.  ,13.  ,  13.  ,  15.  ,  11. , 18.  ,  17.  ,  16.  ,  11.  ,  17.  ,  16.  ,  15.  ,  10.  , 15.  ,  17.  ,  15.  ,  10.  ]
                        ])
    a = rnd.randint(0,5)
    b = rnd.randint(5,10)
    c = rnd.randint(10,15)
    d = rnd.randint(15,20)
    indx = np.array([a,b,c,d])
    Q = Q_master[:, indx].T
    return Q

def get_random_choosen_q(total_num_species=20, num_species = 4):
    rng = rnd.default_rng()
    q_1 = np.concatenate([rng.normal(10,1, size=5),rng.normal(5, 1, size=5),rng.normal(7, 1, size=5),rng.normal(15, 2, size=5)])#speed
    q_2 = np.concatenate([rng.normal(90,5, size=5),rng.normal(150, 9, size=5),rng.normal(120, 10, size=5),rng.normal(75, 12, size=5)]) #footprint
    q_3 = np.concatenate([rng.normal(8,2, size=5),rng.normal(5, 1, size=5),rng.normal(16, 1, size=5),rng.normal(7, 2, size=5)]) #payload
    q_4 = np.concatenate([rng.normal(2,1, size=5),rng.normal(3, 1, size=5),rng.normal(2, 1, size=5),rng.normal(3, 2, size=5)]) #reach
    q_5 = rng.normal(np.random.randint(25,30), 1, size=total_num_species) #weight
    q_6 = rng.normal(np.random.randint(210,230), 15.4, size=total_num_species) #sensing frequency
    q_7 = rng.normal(np.random.randint(58,60), 0.8, size=total_num_species) #sensing range
    q_8 = rnd.choice([0,1,2,3,4], total_num_species) #color
    q_9 = rnd.choice(range(10,20), total_num_species) #battery
    Q_master = np.array([q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9])

    a = rnd.randint(0,5)
    b = rnd.randint(5,10)
    c = rnd.randint(10,15)
    d = rnd.randint(15,20)
    indx = np.array([a,b,c,d])
    Q = Q_master[:, indx].T
    return Q    
