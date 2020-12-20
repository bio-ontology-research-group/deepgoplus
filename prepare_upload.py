from datetime import datetime
import json
import subprocess

data_root = 'data/'
last_release_metadata = 'metadata/last_release.json' 

with open(last_release_metadata, 'r') as f:
    last_release_data = json.load(f)
last_release_date = last_release_data["current_date"]

def compress_data():
    date = datetime.strptime(last_release_data["current_date"], '%a, %d %b %Y %H:%M:%S GMT')
    out_file = "data/data-" + str(date.date()) + '.tar.gz'

    go_file="data/go.obo"
    diamond_db="data/train_data.dmnd"
    model="data/model.h5"
    result_diamond="data/test_diamond.res"

    train_pkl="data/train_data.pkl"
    train_fa="data/train_data.fa"
    test_pkl="data/test_data.pkl"
    test_fa="data/test_data.fa"

    terms="data/terms.pkl"

    release="data/RELEASE.md"

    cmd = ["tar", "-czf", out_file, go_file, diamond_db, model, result_diamond, train_pkl, train_fa, test_pkl, test_fa, terms, release]
#    proc = subprocess.run(cmd)

    return out_file


def metrics_from_files():
    mf = open('results/deepgoplus_mf.txt').readlines()
    bp = open('results/deepgoplus_bp.txt').readlines()
    cc = open('results/deepgoplus_cc.txt').readlines()

    mf_smin = mf[2].split(':')[1]
    mf_fmax = mf[3].split(':')[1]
    mf_aupr = mf[4].split(':')[1]

    bp_smin = bp[2].split(':')[1]
    bp_fmax = bp[3].split(':')[1]
    bp_aupr = bp[4].split(':')[1]

    cc_smin = cc[2].split(':')[1]
    cc_fmax = cc[3].split(':')[1]
    cc_aupr = cc[4].split(':')[1]

    return mf_smin, mf_fmax, mf_aupr, bp_smin, bp_fmax, bp_aupr, cc_smin, cc_fmax, cc_aupr


def release_notes_file():
    with open(last_release_metadata, 'r') as f:
            last_release_data = json.load(f)

    version = last_release_data["version"]

    file = open('data/RELEASE.html', 'w')
    go_file = open('data/go.obo', 'r')
    mf_smin, mf_fmax, mf_aupr, bp_smin, bp_fmax, bp_aupr, cc_smin, cc_fmax, cc_aupr = metrics_from_files()

    go_file.readline()
    go_date =  str(datetime.strptime(go_file.readline().rstrip('\n').split('/')[1], '%Y-%m-%d').date())
    swissprot_date = str(datetime.strptime(last_release_data["current_date"], '%a, %d %b %Y %H:%M:%S GMT').date())

   


    text = """
<!DOCTYPE html>
<html>
    <head>
        <title>Release notes. Version: """ + version +"""</title>
    </head>
    <body>
        <h1>DeepGOPlus: Improved protein function prediction from sequence</h1>
        <p>DeepGOPlus is a novel method for predicting protein functions from
            protein sequences using deep neural networks combined with sequence
            similarity based predictions.</p>
        <h2>Release information</h2>

        <p>

            The model in the current release was trained using the Gene Ontology
            released on """ + go_date + """ and the SwissProt data released on """ + swissprot_date+  """.

            The obtained results are the following:
        </p>

        <ul>
            <li>
                For MFO:
                <ul>
                    <li> Fmax:   """ + mf_fmax +"""</li>
                    <li> Smin:   """ + mf_smin +"""</li>
                    <li> AUPR:   """ + mf_aupr +"""</li>
                </ul>
            </li>
            <li>
                For BPO:
                <ul>
                    <li> Fmax:   """ + bp_fmax + """</li>
                    <li> Smin:   """ + bp_smin + """</li>
                    <li> AUPR:   """ + bp_aupr + """</li>
                </ul>
            </li>
            <li>        
                For CCO:
                <ul>
                    <li> Fmax:   """ + cc_fmax + """</li> 
                    <li> Smin:   """ + cc_smin + """</li>    
                    <li> AUPR:   """ + cc_aupr + """</li>
                </ul>
            </li>
        </ul>

   
        <p>For more information about the project, please look at <a href= "https://github.com/bio-ontology-research-group/deepgoplus" target="_blank">https://github.com/bio-ontology-research-group/deepgoplus</a>  
        </p>
    </body>
</html>
"""
    file.write(text)




def upload_data(filename):
    
    deepgo_server_ip = "10.254.146.187"
    login_cmd = "zhapacfp@"+deepgo_server_ip
    # cmd = ["scp", filename, login_cmd+":"+"/opt-"+filename]
    # proc = subprocess.run(cmd)

    cmd = ["ssh", "-t", login_cmd, "sudo mkdir /opt-data/test"]
    proc = subprocess.run(cmd)



def main():
    release_notes_file()
    out_file_name = compress_data() #compress the data and return the name(string) of the file
    upload_data(out_file_name)

if __name__ == "__main__":
    main()

