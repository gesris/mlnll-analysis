#!/bin/bash
USERNAME=$(whoami)
condor_qedit $USERNAME -constraint 'JobStatus=?=5' NumJobStarts 1
condor_qedit $USERNAME -constraint 'JobStatus=?=5' NumShadowStarts 1
condor_qedit $USERNAME -constraint 'JobStatus=?=5' JobRunCount 1
condor_qedit $USERNAME -constraint 'JobStatus=?=5' NumJobMatches 1
condor_qedit $USERNAME -constraint 'JobStatus=?=5' NumShadowExceptions 1
condor_release $USERNAME
