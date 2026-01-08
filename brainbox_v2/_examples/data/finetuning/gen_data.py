import json
import random

def generate_cyberpunk_data():
    queries = [
        "How do I secure my port 8080?", "Identify the intrusion vector.", "Scan for vulnerabilities in the subnet.",
        "Decrypt this log entry.", "Initialize firewall protocols.", "Bypass the proxy layer.",
        "Audit the access logs.", "Optimize my neural link connection.", "Verify the integrity of the database.",
        "Trace the packet source.", "Detect the malware signature.", "Isolate the infected node.",
        "Refactor the encryption module.", "Establish a secure tunnel.", "Ping the remote server.",
        "List all active processes.", "Kill the rogue process.", "Mount the encrypted volume.",
        "Generate a keyset.", "Renew the security certificates.", "Analyze the traffic spike.",
        "Update the threat database.", "Synchronize with the master node.", "Fetch the audit report.",
        "Reboot the core security unit.", "Calibrate the IDS sensors.", "Configure the load balancer.",
        "Archive the incident report.", "Check the backup status.", "Scan the external hardware."
    ]
    
    prefixes = ["> [SEC_OPS: ALERT]", "> [SEC_OPS: INFO]", "> [SEC_OPS: TRACE]", "> [SEC_OPS: AUDIT]"]
    keywords = ["ICE", "uplink", "grid", "matrix", "protocol", "kernel", "shards", "buffer", "node", "vector"]
    
    dataset = []
    
    # Generate 110 entries to be safe
    for i in range(110):
        query = random.choice(queries) + f" (Instance {i})"
        prefix = random.choice(prefixes)
        kw = random.choice(keywords)
        
        # Structure: Prefix + Status + Technical Slang + Answer
        output = f"{prefix} Status: COMPLETED. {kw.capitalize()} integrity verified. {query.replace(' (Instance ' + str(i) + ')', '')} has been processed via neural-link."
        
        dataset.append({"input": query, "output": output})
        
    with open("data/finetuning/dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {len(dataset)} lines in data/finetuning/dataset.jsonl")

if __name__ == "__main__":
    generate_cyberpunk_data()
