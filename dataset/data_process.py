def data_process(filepath):
    with open(filepath,"r+") as fr:
        data=fr.readlines()
    s = "\t".join(["user_id:token","item_id:token"])
    s+="\n"
    for d in data:
        seq=d.split(" ")
        uid,items=seq[0],list(map(int,seq[1:]))
        for i in items:
            temp_seq="\t".join([uid,str(i)])
            s+=temp_seq
            s+="\n"
    with open(filepath[:-4]+".inter", "w+") as ff:
        ff.write(s)

if __name__ == '__main__':
    import os
    filenames=os.listdir(os.getcwd())
    for f in filenames:
        if os.path.isdir(f):
            file_path="./"+f+"/"+f+".txt"
            data_process(file_path)
            print(f)


