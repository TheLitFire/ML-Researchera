#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<time.h>

unsigned char data[40000][32*32*3];
int dis[10000][40000];
unsigned char label[10000][40000];
unsigned char anslabel[10000];
int totlabel[10];
unsigned char testdt[10000][32*32*3];
unsigned char all_data[50000][32*32*3];
unsigned char all_label[50000];
_Bool is_test[50000];

void q_sort(int dis[], unsigned char label[], int l, int r){
    int i = l, j = r, m = dis[l + r >> 1];
    while (i <= j){
        while (dis[i] < m) ++i;
        while (dis[j] > m) --j;
        if (i <= j){
            int temp = dis[i];
            dis[i] = dis[j];
            dis[j] = temp;
            unsigned char t = label[i];
            label[i] = label[j];
            label[j] = t;
            ++i; --j;
        }
    }
    if (i < r) q_sort(dis, label, i, r);
    if (l < j) q_sort(dis, label, l, j);
}

/*inline int distance(int a, int b){
    return (a > b ? a - b : b - a);
}*/
inline int distance(int a, int b){
    return (a - b) * (a - b);
}

void generate_flag(void){
    srand(time(0));
    memset(is_test, 0, sizeof(is_test));
    int total = 0;
    while (total < 10000){
        int x = rand()%50000;
        if (is_test[x]) continue;
        is_test[x] = 1;
        ++total;
    }
}

void read_all_data(void){
    FILE *ifp = fopen("train_data.dat", "rb");
    assert(fread(all_data[0], 1, 50000*32*32*3, ifp) == 50000*32*32*3);
    unsigned char junk;
    assert(fread(&junk, 1, 1, ifp) == 0);
    fclose(ifp);
    ifp = fopen("train_label.dat", "rb");
    assert(fread(all_label, 1, 50000, ifp) == 50000);
    assert(fread(&junk, 1, 1, ifp) == 0);
    fclose(ifp);
}

void get_data(void){
    generate_flag();
    int tot_train = 0, tot_test = 0;
    for (int i = 0; i < 50000; ++i){
        if (is_test[i]){
            memcpy(testdt[tot_test], all_data[i], sizeof(testdt[tot_test]));
            anslabel[tot_test++] = all_label[i];
        }else{
            memcpy(data[tot_train], all_data[i], sizeof(data[tot_train]));
            label[0][tot_train++] = all_label[i];
        }
    }
    assert(tot_test == 10000 && tot_train == 40000);
    for (int i = 1; i < 10000; ++i)
        memcpy(label[i], label[0], sizeof label[i]);
}

int main(void){
    read_all_data();
    for (;;){
        get_data();
        for (int i = 0; i < 10000; ++i){
            for (int j = 0; j < 40000; ++j){
                dis[i][j] = 0;
                for (int k = 0; k < 32*32*3; ++k)
                    dis[i][j] += distance((int)testdt[i][k],(int)data[j][k]);
            }
            //printf("dis calc finish: %d\n",i);
        }

        for (int i = 0; i < 10000; ++i){
            q_sort(dis[i], label[i], 0, 39999);
            //printf("sort finish: %d\n",i);
        }
        
        int bestk = 1, mintotalmis = 1000000;
        for (int k = 1; k < 100; ++k){
            int totalmis = 0;
            for (int i = 0; i < 10000; ++i){
                memset(totlabel, 0, sizeof totlabel);
                for (int j = 0; j < k; ++j)
                    ++totlabel[label[i][j]];
                int maxx = 0;
                for (int j = 1; j < 10; ++j)
                    if (totlabel[maxx] < totlabel[j])
                        maxx = j;
                if (maxx != (int)anslabel[i])
                    ++totalmis;
            }
            //printf("k = %d, totalmis = %d\n", k, totalmis);
            if (totalmis < mintotalmis){
                bestk = k;
                mintotalmis = totalmis;
            }
        }
        printf("bestk = %d, bestmis = %d\n", bestk, mintotalmis);
    }
}