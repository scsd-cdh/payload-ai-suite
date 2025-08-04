#!/bin/bash

echo "Cleaning Zone.Identifier files from data directory..."
find data -name "*:Zone.Identifier" -exec rm -f {} \;
echo "Data cleaning completed!"