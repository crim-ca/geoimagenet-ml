#!/usr/bin/env groovy

pipeline {
    agent any

    environment {
        TAG_NAME = sh(returnStdout: true, script: '[[ -z $(git tag -l --points-at HEAD) ]] && printf latest || printf $(git tag -l --points-at HEAD)')
        LOCAL_IMAGE_NAME = "geoimagenet_ml:$TAG_NAME"
        LATEST_IMAGE_NAME = "docker-registry.crim.ca/geoimagenet/ml:latest"
        TAGGED_IMAGE_NAME = "docker-registry.crim.ca/geoimagenet/ml:$TAG_NAME"
    }

    options {
        buildDiscarder (logRotator(numToKeepStr:'10'))
    }

    stages {

        stage('Build') {
            steps {
                sh 'env | sort'
                sh 'docker build -t $LOCAL_IMAGE_NAME .'
            }
        }

        stage('Test') {
            steps {
                script {
                    docker.image('mongo:3.4.0').withRun('-e "ALLOW_IP_RANGE=0.0.0.0/0" -e "IP_LIST=*"') { c ->
                        sh """
                        docker run --rm --link ${c.id}:mongodb -e MONGODB_HOST=mongodb -e MONGODB_PORT=27017 $LOCAL_IMAGE_NAME /bin/sh -c \" \
                        TEST_MODEL_URL="https://geoimagenetdev.crim.ca/ml/models/2491b763-c785-42b9-83c1-78cabfe51bbc/download" make test-req test-unit"
                        """
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                anyOf {
                    environment name: 'GIT_LOCAL_BRANCH', value: 'release';
                    not { environment name: 'TAG_NAME', value: 'latest' }
                }
            }
            steps {
                sh 'docker tag $LOCAL_IMAGE_NAME $TAGGED_IMAGE_NAME'
                sh 'docker push $TAGGED_IMAGE_NAME'
                sh 'docker tag $LOCAL_IMAGE_NAME $LATEST_IMAGE_NAME'
                sh 'docker push $LATEST_IMAGE_NAME'
                sh 'ssh ubuntu@geoimagenetdev.crim.ca "cd ~/compose && ./geoimagenet-compose.sh down && ./geoimagenet-compose.sh pull && ./geoimagenet-compose.sh up -d"'
                slackSend channel: '#geoimagenet-dev', color: 'good', message: "*GeoImageNet ML*:\nPushed docker image: `${env.TAGGED_IMAGE_NAME}`\nDeployed to: https://geoimagenetdev.crim.ca/ml"
            }
        }
    }
    post {
       success {
           slackSend channel: '#geoimagenet-dev', color: 'good', message: "*GeoImageNet ML*: Build #${env.BUILD_NUMBER} *successful* on git branch `${env.GIT_LOCAL_BRANCH}` :tada: (<${env.BUILD_URL}|View>)"
       }
       failure {
           slackSend channel: '#geoimagenet-dev', color: 'danger', message: "*GeoImageNet ML*: Build #${env.BUILD_NUMBER} *failed* on git branch `${env.GIT_LOCAL_BRANCH}` :sweat_smile: (<${env.BUILD_URL}|View>)"
       }
    }
}
