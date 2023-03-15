#!/usr/bin/env bats

# Smoke Test for NER container
# These tests expect a running container at port 8001 named tp-ner

@test "logs show clean start" {
  result=$(docker logs tp-ner | grep -o 'Application startup complete.')
  [ "$result" == 'Application startup complete.' ]
}