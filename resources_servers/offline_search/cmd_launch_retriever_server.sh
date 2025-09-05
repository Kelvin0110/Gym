#!/bin/bash

source /lustre/fsw/portfolios/llmservice/users/rgala/repos/ASearcher/.venv/bin/activate


#These indices and corpus are on dfw 
python resources_servers/offline_search/retriever_server.py \
--index_path /lustre/fsw/portfolios/llmservice/users/rgala/repos/ASearcher/temp_ritu_v2/pj_index/e5_Flat.index \
--corpus_path '/lustre/fsw/portfolios/llmservice/users/rgala/repos/ASearcher/temp_ritu_v2/pj_index/wiki-18.jsonl' \
--retriever_name e5 \
--retriever_model intfloat/e5-base-v2 \
--faiss_gpu \
--topk 10 