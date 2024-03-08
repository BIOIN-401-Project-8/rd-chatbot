# Rare Genetic Diseases Chatbot

A chatbot that can knowledgably extract literature from PubMed and inform people
about the latest information and updates regarding rare genetic diseases.
This chatbot is a BIOIN 401 Project at the University of Alberta.

## Setup
```bash
docker compose up -d app neo4j ollama
```

Note, to setup Dynamic DNS with Namecheap, add the following line to your crontab with `crontab -e`:
```bash
0 * * * * cd ~/Github/bioin-401-project/rgd-chatbot && docker compose run namecheap-ddns
```

### Reverse Proxy
To setup the reverse proxy on the Wishart lab server (which is behind a VPN),
run the following command on a machine that has a public IP address:
```bash
docker compose up -d nginx-proxy-manager rathole-server
```
and then run the following command on the Wishart lab server.
```bash
docker compose up -d rathole-client
```


## People
Team: [Steven Tang](https://github.com/steventango) and [Robyn Woudstra](https://github.com/rwoudstr)

Mentors: Mark Berjanskii, Vasuk Gautam, Scott MacKay, Mahi Zakir

Instructors: [Dr. David Wishart](https://www.wishartlab.com/members/david-wishart), [Dr. Gane Wong](https://sites.google.com/a/ualberta.ca/professor-gane-ka-shu-wong/)
