import groovy.xml.MarkupBuilder
// Manages communication with github for reporting important status
// events. Relies on Github integration plugin https://github.com/jenkinsci/github-plugin
class GithubStatus {

    // Repository to send status e.g: '/username/repo'
    String Repository

    // Commit hash to send status e.g: `909b76f97e`
    String Commit_id

    // Build url send status e.g: `https://jenkins/build/123`
    String BUILD_URL

    // Set pending status for this pull request
    def setPending(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "In Progress", "PENDING", result_status_when_set_commit_failure)
    }

    // Set failed status for this pull request
    def setFailed(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "Complete", "FAILURE", result_status_when_set_commit_failure)
    }

    // Set success status for this pull request
    def setSuccess(script, String context_name, String result_status_when_set_commit_failure = "UNSTABLE") {
        this.setBuildStatusStep(script, context_name, "Complete", "SUCCESS", result_status_when_set_commit_failure)
    }

    // Manually set PR status via GitHub integration plugin. Manual status updates are needed
    // Due to pipeline stage retries not propagating updates back to Github automatically
    def setBuildStatusStep(script, String context_name, String message, String state, String result_status_when_set_commit_failure) {
        def commitContextName = context_name
        def build_url_secret = this.BUILD_URL.replace("icl-jenkins.sc", "llvm-ci")

        script.retry(4) {
            script.step([
                    $class             : "GitHubCommitStatusSetter",
                    reposSource        : [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/${this.Repository}"],
                    contextSource      : [$class: "ManuallyEnteredCommitContextSource", context: commitContextName],
                    errorHandlers      : [],
                    commitShaSource    : [$class: "ManuallyEnteredShaSource", sha: this.Commit_id],
                    //statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${this.BUILD_URL}flowGraphTable/"],
                    statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: "${build_url_secret}"],
                    statusResultSource : [$class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]]]
            ])
        }


    }
}


def fill_task_name_description () {
    script {
        currentBuild.displayName = "PR#${env.PR_number}-No.${env.BUILD_NUMBER}"
        currentBuild.description = "PR number: ${env.PR_number} / Commit id: ${env.Commit_id}"
    }
}

def githubStatus = new GithubStatus(
        Repository: params.Repository,
        Commit_id: params.Commit_id,
        BUILD_URL: env.RUN_DISPLAY_URL
)

def shell(String command, String label_string = "Bat Command") {
    return bat(returnStdout: true, script: "sh -x -c \"${command}\"", label: label_string).trim()
}


build_ok = true
fail_stage = ""
user_in_github_group = false

pipeline {

    agent { label "master" }
    options {
        durabilityHint 'PERFORMANCE_OPTIMIZED'
        timeout(time: 5, unit: 'HOURS')
        timestamps()
    }

    environment {
        def NUMBER = sh(script: "expr ${env.BUILD_NUMBER}", returnStdout: true).trim()
        def TIMESTEMP = sh(script: "date +%s", returnStdout: true).trim()
        def DATESTEMP = sh(script: "date +\"%Y-%m-%d\"", returnStdout: true).trim()
    }

    parameters {
        string(name: 'Commit_id', defaultValue: 'None', description: '',)
        string(name: 'PR_number', defaultValue: 'None', description: '',)
        string(name: 'Repository', defaultValue: 'oneapi-src/oneDPL', description: '',)
        string(name: 'User', defaultValue: 'None', description: '',)
    }
    triggers {
        GenericTrigger(
                genericVariables: [
                        [key: 'Commit_id', value: '$.pull_request.head.sha', defaultValue: 'None'],
                        [key: 'PR_number', value: '$.number', defaultValue: 'None'],
                        [key: 'Repository', value: '$.pull_request.base.repo.full_name', defaultValue: 'None'],
                        [key: 'User', value: '$.pull_request.user.login', defaultValue: 'None'],
                        [key: 'action', value: '$.action', defaultValue: 'None']
                ],

                causeString: 'Triggered on $PR_number',

                token: 'oneDPL-pre-ci',

                printContributedVariables: true,
                printPostContent: true,

                silentResponse: false,

                regexpFilterText: '$action',
                regexpFilterExpression: '(opened|reopened|synchronize)'
        )
    }

    stages {
        stage('Check_User_in_Org') {
            agent {
                label "master"
            }
            steps {
                script {
                    try {
                        retry(2) {
                            fill_task_name_description()
                            def check_user_return = sh(script: "python3 /localdisk2/oneDPL_CI/check_user_in_group.py -u  ${env.User}", returnStatus: true, label: "Check User in Group")
                            echo "check_user_return value is $check_user_return"
                            if (check_user_return == 0) {
                                user_in_github_group = true
                            }
                            else {
                                user_in_github_group = false
                                currentBuild.result = 'UNSTABLE'
                            }
                        }
                    }
                    catch (e) {
                        fail_stage = fail_stage + "    " + "Check_User_in_Org"
                        user_in_github_group = false
                        echo "Exception occurred when check User:${env.User} in group. Will skip build this time"
                        sh script: "exit -1", label: "Set Failure"
                    }
                }
            }
        }

        stage('Win_Check') {
            when {
                expression { user_in_github_group }
            }
            agent { label "windows_test" }
            stages {
                stage('Git-monorepo'){
                    steps {
                        script {
                            try {
                                retry(2) {
                                    githubStatus.setPending(this, "Jenkins/Win_Check")
                                    deleteDir()

                                    checkout changelog: false, poll: false, scm: [$class: 'GitSCM', branches: [[name: '${Commit_id}']], doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'src'], [$class: 'CloneOption', timeout: 200]], submoduleCfg: [], userRemoteConfigs: [[refspec: "+refs/pull/${PR_number}/head:refs/remotes/origin/PR-${PR_number}", url: 'https://github.com/oneapi-src/oneDPL.git']]]

                                }
                            }
                            catch (e) {
                                build_ok = false
                                fail_stage = fail_stage + "    " + "Git-monorepo"
                                bat "exit -1"
                            }
                        }
                    }
                }

                stage('Setting_Env') {
                    steps {
                        script {
                            output = bat returnStdout: true, script: '@call "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat" > NUL && set'
                            oneapi_env = output.split('\r\n') as List
                        }
                    }
                }

                stage('Check_Samples'){
                    steps {
                        timeout(time: 1, unit: 'HOURS'){
                            script {
                                try {
                                    bat script: """
                                            md oneAPI-samples
                                            xcopy D:\\netbatch\\iusers\\oneDPL_CI\\oneAPI-samples .\\oneAPI-samples /E /Q /H
                                            cd oneAPI-samples
                                            git pull origin master
                                            
                                        """, label: "Prepare oneAPI-samples"

                                    try {
                                        withEnv(oneapi_env) {
                                            bat script: """
                                                cmd.exe /c
                                                call "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Visual Studio 2017\\Visual Studio Tools\\VC\\x64 Native Tools Command Prompt for VS 2017.lnk"  
                                                call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64  
                                                
                                                d:
                                                cd ${env.WORKSPACE}\\oneAPI-samples\\Libraries\\oneDPL\\gamma-correction
                                                
                                                MSBuild gamma-correction.sln /t:Rebuild /p:Configuration="Release"
                                            """, label: "Gamma_return_value Test Step"
                                        }
                                    }
                                    catch(e) {
                                        build_ok = false
                                        fail_stage = fail_stage + "    " + "Check_Samples_gamma-correction"
                                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                            bat 'exit 1'
                                        }
                                    }

                                    try {
                                        withEnv(oneapi_env) {
                                            bat script: """
                                                cmd.exe /c
                                                call "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Visual Studio 2017\\Visual Studio Tools\\VC\\x64 Native Tools Command Prompt for VS 2017.lnk"                                    
                                                call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64  
                                                
                                                d:
                                                cd ${env.WORKSPACE}\\oneAPI-samples\\Libraries\\oneDPL\\stable_sort_by_key\\src
                                                echo "Build&Test command: dpcpp /W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /Isrc/include main.cpp -o test.exe && test.exe"
                                                dpcpp /W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /I${env.WORKSPACE}/src/include main.cpp -o test.exe && test.exe
                                            """, label: "Stable_sort_by_key Test Step"
                                        }
                                    }
                                    catch(e) {
                                        build_ok = false
                                        fail_stage = fail_stage + "    " + "Check_Samples_stable_sort_by_key"
                                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                            bat 'exit 1'
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    fail_stage = fail_stage + "    " + "Check_Samples"
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Check_pstl_testsuite'){
                    steps {
                        timeout(time: 2, unit: 'HOURS'){
                            script {
                                def results = []
                                try {
                                    script {
                                        def tests = findFiles glob: 'src/test/pstl_testsuite/**/*pass.cpp' //uncomment this line to run all tests
                                        echo tests.toString()
                                        def failCount = 0
                                        def passCount = 0

                                        withEnv(oneapi_env) {
                                            for ( x in tests ) {
                                                try {
                                                    phase = "Build&Run"
                                                    bat script: """
                                                        dpcpp /W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /Isrc/include /Isrc/test/pstl_testsuite $x
                                                        test.exe
                                                    """, label: "Check $x"
                                                    passCount++
                                                    results.add([name: x, pass: true, phase: phase])
                                                }
                                                catch (e) {
                                                    failCount++
                                                    results.add([name: x, pass: false, phase: phase])
                                                }
                                            }
                                        }
                                        xml = write_results_xml(results)
                                        echo "Passed tests: $passCount, Failed tests: $failCount"
                                        if (failCount > 0) {
                                            bat 'exit 1'
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    fail_stage = fail_stage + "    " + "Check_Run"
                                    failed_cases = "Failed cases are: "
                                    results.each { item ->
                                        if (!item.pass) {
                                            failed_cases = failed_cases + "\n" + "${item.name}"
                                        }
                                    }
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Check_extensions_testsuite'){
                    steps {
                        timeout(time: 2, unit: 'HOURS'){
                            script {
                                def results = []
                                try {
                                    script {
                                        //def tests = findFiles glob: 'ci/test/pstl_testsuite/pstl/**/*pass.cpp'
                                        def tests = findFiles glob: 'src/test/extensions_testsuite/**/*pass.cpp' //uncomment this line to run all tests
                                        echo tests.toString()
                                        def failCount = 0
                                        def passCount = 0

                                        withEnv(oneapi_env) {
                                            for ( x in tests ) {
                                                try {
                                                    phase = "Build&Run"
                                                    bat script: """
                                                        dpcpp /W0 /nologo /D _UNICODE /D UNICODE /Zi /WX- /EHsc /Fetest.exe /Isrc/include /Isrc/test/extensions_testsuite $x
                                                        test.exe
                                                    """, label: "Check $x"
                                                    passCount++
                                                    results.add([name: x, pass: true, phase: phase])
                                                }
                                                catch (e) {
                                                    failCount++
                                                    results.add([name: x, pass: false, phase: phase])
                                                }
                                            }
                                        }
                                        xml = write_results_xml(results)
                                        echo "Passed tests: $passCount, Failed tests: $failCount"
                                        if (failCount > 0) {
                                            bat 'exit 1'
                                        }
                                    }
                                }
                                catch(e) {
                                    build_ok = false
                                    fail_stage = fail_stage + "    " + "Check_extensions_testsuite"
                                    failed_cases = "Failed cases are: "
                                    results.each { item ->
                                        if (!item.pass) {
                                            failed_cases = failed_cases + "\n" + "${item.name}"
                                        }
                                    }
                                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                        bat 'exit 1'
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    post {
        always {
            script {
                if (user_in_github_group) {
                    if (build_ok) {
                        currentBuild.result = "SUCCESS"
                        githubStatus.setSuccess(this, "Jenkins/Win_Check")
                    } else {
                        currentBuild.result = "FAILURE"
                        githubStatus.setFailed(this, "Jenkins/Win_Check")
                    }
                }
           }
        }
    }

}
@NonCPS
def write_results_xml(results) {
    def xmlWriter = new StringWriter()
    def xml = new MarkupBuilder(xmlWriter)
    xml.testsuites{
        delegate.testsuite(name: 'tests') {
            results.each { item ->
                if (item.pass) {
                    delegate.delegate.testcase(name: item.name, classname: item.name)
                }
                else {
                    delegate.delegate.testcase(name: item.name, classname: item.name) {
                        delegate.failure(message: 'Fail', item.phase)
                    }
                }
            }
        }
    }
    return xmlWriter.toString()
}