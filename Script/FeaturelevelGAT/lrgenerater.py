

LRfilepath = "./Knowledge/CellChat.csv"
newLRfilepath = "./Knowledge/CellChat_split.csv"

title = True
with open(LRfilepath,"r") as LRfile:
    with open(newLRfilepath,"w") as newLRfile:
        for line in LRfile.readlines():
            if title:
                newLRfile.write(line)
                title = False
            else:
                linedata = line.strip().split(",")
                ligands = linedata[0].split("_")
                receptors = linedata[1].split("_")
                for ligand in ligands:
                    for receptor in receptors:
                        pline = ligand.upper()+','+receptor.upper()+"\n"
                        newLRfile.write(pline)



