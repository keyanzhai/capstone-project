import datetime
from test1 import test1
from test2 import test2, test2_pass
from test3 import test3
from test3OpticalFlow import test3_optical_flow
from test3b import test3b
from vis import vis_directions

if __name__ == "__main__":
    report_file = open('../report/report.txt', 'w')
    report_file.write("Report for fall risk assessment\n\n")
    
    at_risk = False
    total_tests = 3
    passed_tests = 0

    # =========================== gather user info ===========================
    report_file.write("-------------------------------\n")
    report_file.write("User information\n\n")
    
    print("Please enter user name:")
    user_name = str(input())
    report_file.write("Name: %s\n" % user_name)

    print("Please enter user age:")
    user_age = int(input())
    report_file.write("Age: %d\n" % user_age)

    print("Please enter user gender (male or female):")
    user_gender = str(input())
    while (user_gender != "male" and user_gender != "female"):
        print("Error input. Please enter user gender (male or female):")
        user_gender = str(input())
    report_file.write("Gender: %s\n" % user_gender)

    print("Please enter user height in cm:")
    user_height = int(input())
    report_file.write("Height: %d cm\n" % user_height)

    now = datetime.datetime.now()
    report_file.write("Test date: %s\n" % now.strftime("%Y-%m-%d"))

    # ============================== Run test1 (test_idx = 1) =================================
    print("\nRunning test1 - the TUG test")
    report_file.write("\n-------------------------------\n")
    report_file.write("Test1 - Timed Up & Go (TUG)\n\n")

    test_idx = 1
    vis_directions(test_idx)
    test1_time = test1()

    print("test1 finished\n")
    report_file.write("Total time = %.2f seconds. \n\n" % test1_time)

    if (test1_time >= 12):
        # User is at risk for falling
        at_risk = True
        report_file.write("Test1 failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("Test1 passed. User is not at risk for falling.\n")
    # ==========================================================================

    # ============================== Run test2 (test_idx = 2) =================================
    print("Running test2 - 30-Second Chair Stand")
    report_file.write("\n-------------------------------\n")
    report_file.write("Test2 - 30-Second Chair Stand\n\n")
    
    test_idx = 2
    vis_directions(test_idx)
    test2_count = test2()
    
    print("test2 finished\n")
    report_file.write("Stand count = %d.\n\n" % test2_count)

    if (test2_pass(user_age, user_gender, test2_count) == False):
        # User is at risk for falling
        at_risk = True
        report_file.write("Test2 failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("Test2 passed. User is not at risk for falling.\n")
    # ==========================================================================

    # ============================== Run test3 =================================
    print("Running test3 - 4-Stage Balance Test\n")
    report_file.write("\n-------------------------------\n")
    report_file.write("Test3 - 4-Stage Balance Test\n")

    # ============================== test3-stage1 (test_idx = 3, stage_idx = 1) ==============================
    report_file.write("\n\t-------------------------------\n")
    report_file.write("\tTest3 stage1 - stand with your feet side-by-side\n\n")
    print("\ttest3 stage1 - stand with your feet side-by-side.")

    test_idx = 3
    stage_idx = 1
    vis_directions(test_idx)
    test3_stage1_time = test3_optical_flow(stage_idx)
    report_file.write("\tTime = %.2f seconds.\n\n" % test3_stage1_time)
    print("\ttest3 stage1 finished\n")
    
    if (test3_stage1_time < 10):
        # test failed
        at_risk = True
        report_file.write("\tTest3 stage1 failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("\tTest3 stage1 passed. User is not at risk for falling.\n")
    # ==================================================================================

    # ============================== test3-stage2-left (test_idx = 4, stage_idx = 2) ========================
    report_file.write("\n\t-------------------------------\n")
    report_file.write("\tTest3 stage2-left - place the instep of left foot so it is touching the big toe of the right foot\n\n")
    print("\ttest3 stage2-left - place the instep of left foot so it is touching the big toe of the right foot.")

    test_idx = 4
    stage_idx = 2
    vis_directions(test_idx)
    test3_stage2_left_time = test3_optical_flow(stage_idx)
    report_file.write("\tTime = %.2f seconds.\n\n" % test3_stage2_left_time)
    print("\ttest3 stage2-left finished\n")

    if (test3_stage2_left_time < 10):
        # test failed
        at_risk = True
        report_file.write("\tTest3 stage2-left failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("\tTest3 stage2-left passed. User is not at risk for falling.\n")
    # ==================================================================================

    # ============================== test3-stage3-left (test_idx = 5, stage_idx = 3) ========================
    report_file.write("\n\t-------------------------------\n")
    report_file.write("\tTest3 stage3-left - Tandem stand: Place left foot in front of the right foot, heel touching toe. \n\n")
    print("\ttest3 stage3-left - Tandem stand: Place left foot in front of the right foot, heel touching toe.")

    test_idx = 5
    stage_idx = 3
    vis_directions(test_idx)
    test3_stage3_left_time = test3_optical_flow(stage_idx)
    report_file.write("\tTime = %.2f seconds.\n\n" % test3_stage3_left_time)
    print("\ttest3 stage3-left finished\n")

    if (test3_stage3_left_time < 10):
        # test failed
        at_risk = True
        report_file.write("\tTest3 stage3-left failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("\tTest3 stage3-left passed. User is not at risk for falling.\n")
    # ==================================================================================

    # ============================== test3-stage4-right (test_idx = 6) =======================
    report_file.write("\n\t-------------------------------\n")
    report_file.write("\tTest3 stage4-right - Stand on right foot.\n\n")
    print("\ttest3 stage4-right - Stand on right foot.")

    test_idx = 6
    vis_directions(test_idx)
    test3_stage4_right_time = test3b()
    report_file.write("\tTime = %.2f seconds.\n\n" % test3_stage4_right_time)
    print("\ttest3 stage4-right finished\n")

    if (test3_stage4_right_time < 10):
        # test failed
        at_risk = True
        report_file.write("\tTest3 stage4-right failed. User is at risk for falling.\n")
    else:
        passed_tests += 1
        report_file.write("\tTest3 stage4-right passed. User is not at risk for falling.\n")
    # ==================================================================================

    print("\ntest3 finished\n")
    print("\nAll tests finished\n")
    report_file.write("\n-------------------------------\n")
    report_file.write("Summary\n\n")

    if (at_risk):
        report_file.write("User is at risk for falling.\n\n")
    else:
        report_file.write("User is not at risk for falling.\n\n")
    
    # report_file.write("Passed %d out of %d tests.\n" % (passed_tests, total_tests))


    report_file.close()




