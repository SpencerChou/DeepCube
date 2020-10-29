var state = [];
var rotateIdxs_old = null;
var rotateIdxs_new = null;
var stateToFE = null;
var FEToState = null;
var legalMoves = null;

var solveStartState = [];
var solveMoves = [];
var solveMoves_rev = [];
var solveIdx = null;
var solution_text = null;

var faceNames = ["top", "bottom", "left", "right", "back", "front"];
var colorMap = {
    0: "#ffffff",
    1: "#ffff1a",
    4: "#0000ff",
    5: "#33cc33",
    2: "#ff8000",
    3: "#e60000"
};
var lastMouseX = 0,
lastMouseY = 0;
var rotX = -30,
rotY = -30;

var moves = []

function init(way="scramble")
{
	if(way == "scramble")
	{
		//tempstate中保存的初始状态的状态，应该是demo_input.json中的序列
		var tempstate = {0: 33,1: 43,2: 2,3: 37,4: 4,5: 39,6: 8,7: 50,8: 51,9: 18,10: 52,11: 6,12: 7,13: 13,14: 41,15: 26,16: 16,17: 11,18: 36,19: 34,20: 35,21: 25,22: 22,23: 21,24: 47,25: 32,26: 9,27: 27,28: 12,29: 44,30: 48,31: 31,32: 1,33: 53,34: 5,35: 24,36: 0,37: 30,38: 45,39: 46,40: 40,41: 23,42: 15,43: 19,44: 20,45: 38,46: 3,47: 17,48: 10,49: 49,50: 14,51: 42,52: 28,53: 29};
		// var tempstate =[6,52,44,32,4,34,35,30,9,33,5,0,19,13,25,11,1,51,15,3,2,41,22,28,47,39,53,17,21,8,23,31,16,24,48,18,45,14,42,43,40,37,36,50,20,26,12,29,46,49,7,27,10,38];
        var newState = new Array(54)
		for(i =0; i<54; i++)
			newState[i] = tempstate[i];
		return newState;
	}
	else{
		 //solveMoves1保存的是还原的步骤，应该对应的demo_output.json文件中的序列。demo_output.json也有可能对应solveMoves_rev1，都试一下
		 //有个前提是solveMoves1和solveMoves_rev1必须是相反的，因为一个是解，一个是还原，solution_text改或者不改变都行。
		 var solveMoves1 = ['D_1', 'L_-1', 'U_-1', 'R_-1', 'F_-1', 'U_1', 'B_1', 'F_-1', 'R_-1', 'U_-1', 'R_1', 'L_-1', 'B_1', 'R_-1', 'R_-1', 'B_-1', 'R_1', 'B_1', 'R_1', 'D_-1', 'R_1', 'B_-1', 'D_1', 'R_1', 'D_-1', 'R_-1', 'D_-1', 'F_1', 'U_-1', 'F_-1', 'D_1', 'F_1', 'U_1', 'F_-1'];
		 //{0: "U_-1",1: "R_-1",2: "U_-1",3: "L_-1",4: "F_-1",5: "L_-1",6: "D_-1",7: "L_-1",8: "D_-1",9: "B_1",10: "R_1",11: "D_-1",12: "F_1",13: "D_-1",14: "D_-1",15: "B_-1",16: "F_-1",17: "L_1",18: "U_1",19: "L_-1",20: "F_1",21: "L_1",22: "U_-1",23: "L_-1",24: "B_1",25: "L_1"};
		 var solveMoves_rev1 = ['D_-1', 'L_1', 'U_1', 'R_1', 'F_1', 'U_-1', 'B_-1', 'F_1', 'R_1', 'U_1', 'R_-1', 'L_1', 'B_-1', 'R_1', 'R_1', 'B_1', 'R_-1', 'B_-1', 'R_-1', 'D_1', 'R_-1', 'B_1', 'D_-1', 'R_-1', 'D_1', 'R_1', 'D_1', 'F_-1', 'U_1', 'F_1', 'D_-1', 'F_-1', 'U_-1', 'F_1'];
		 //{0: "U_1",1: "R_1",2: "U_1",3: "L_1",4: "F_1",5: "L_1",6: "D_1",7: "L_1",8: "D_1",9: "B_-1",10: "R_-1",11: "D_1",12: "F_-1",13: "D_1",14: "D_1",15: "B_1",16: "F_1",17: "L_-1",18: "U_-1",19: "L_1",20: "F_-1",21: "L_-1",22: "U_1",23: "L_1",24: "B_-1",25: "L_-1"};
		 var solution_text1=["D'", 'L', 'U', 'R', 'F', "U'", "B'", 'F', 'R', 'U', "R'", 'L', "B'", 'R', 'R', 'B', "R'", "B'", "R'", 'D', "R'", 'B', "D'", "R'", 'D', 'R', 'D', "F'", 'U', 'F', "D'", "F'", "U'", 'F'];
		 //{0: "U'",1: "R'",2: "U'",3: "L'",4: "F'",5: "L'",6: "D'",7: "L'",8: "D'",9: "B",10: "R",11: "D'",12: "F",13: "D'",14: "D'",15: "B'",16: "F'",17: "L",18: "U",19: "L'",20: "F",21: "L",22: "U'",23: "L'",24: "B",25: "L"};
		 var lenth = 0;
		 for(var key in solution_text1)
		 {
			 lenth += 1;
		 }
		 solveMoves=[];
		 solveMoves_rev=[];
		 solution_text=[];
		 for(var i=0; i<lenth; i++)
		 {
			 solveMoves.push(solveMoves1[i]);
			 solveMoves_rev.push(solveMoves_rev1[i]);
			 solution_text.push(solution_text1[i]);
		 }
		 
	}
	
}

function reOrderArray(arr, indecies) {
    var temp = []
    for (var i = 0; i < indecies.length; i++) {
        var index = indecies[i]
            temp.push(arr[index])
    }
	//console.log(temp);
    return temp;

}
/*
Rand int between min (inclusive) and max (exclusive)
 */
function randInt(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

//将魔方的六个面全部删除
function clearCube() {
    for (i = 0; i < faceNames.length; i++) {
        var myNode = document.getElementById(faceNames[i]);
        while (myNode.firstChild) {
            myNode.removeChild(myNode.firstChild);
        }
    }
}

function setStickerColors(newState) {
    state = newState
        clearCube()

        idx = 0
        for (i = 0; i < faceNames.length; i++) {
            for (j = 0; j < 9; j++) {
                var iDiv = document.createElement('div');
                iDiv.className = 'sticker';
                iDiv.style["background-color"] = colorMap[Math.floor(newState[idx] / 9)]
                    document.getElementById(faceNames[i]).appendChild(iDiv);
                idx = idx + 1
            }
        }
}

function buttonPressed(ev) {
    var face = ''
        var direction = '1'

        if (ev.shiftKey) {
            direction = '-1'
        }
        if (ev.which == 85 || ev.which == 117) {
            face = 'U'
        } else if (ev.which == 68 || ev.which == 100) {
            face = 'D'
        } else if (ev.which == 76 || ev.which == 108) {
            face = 'L'
        } else if (ev.which == 82 || ev.which == 114) {
            face = 'R'
        } else if (ev.which == 66 || ev.which == 98) {
            face = 'B'
        } else if (ev.which == 70 || ev.which == 102) {
            face = 'F'
        }
        if (face != '') {
            clearSoln();
            moves.push(face + "_" + direction);
            nextState();
        }
}

function enableScroll() {
    document.getElementById("first_state").disabled = false;
    document.getElementById("prev_state").disabled = false;
    document.getElementById("next_state").disabled = false;
    document.getElementById("last_state").disabled = false;
}

function disableScroll() {
    //同样解除各个功能
    document.getElementById("first_state").blur(); //so keyboard input can work without having to click away from disabled button
    document.getElementById("prev_state").blur(); //blur 就是使得下拉列表失去聚焦作用
    document.getElementById("next_state").blur();
    document.getElementById("last_state").blur();

    document.getElementById("first_state").disabled = true;
    document.getElementById("prev_state").disabled = true;
    document.getElementById("next_state").disabled = true;
    document.getElementById("last_state").disabled = true;
}

/*
Clears solution as well as disables scroll
 */
function clearSoln() {
    solveIdx = 0;
    solveStartState = [];
    solveMoves = [];
    solveMoves_rev = [];
    solution_text = null;
    document.getElementById("solution_text").innerHTML = "Solution:";
    disableScroll();
}

function setSolnText(setColor = true) {
    solution_text_mod = JSON.parse(JSON.stringify(solution_text))
        if (solveIdx >= 0) {
            if (setColor == true) {
                solution_text_mod[solveIdx] = solution_text_mod[solveIdx].bold().fontcolor("blue")
            } else {
                solution_text_mod[solveIdx] = solution_text_mod[solveIdx]
            }
        }
        document.getElementById("solution_text").innerHTML = "Solution: " + solution_text_mod.join(" ");
}
//这个keypress会被作为参数传递给buttoPressed函数

function enableInput() {
    document.getElementById("scramble").disabled = false;
    document.getElementById("solve").disabled = false;
    $(document).on("keypress", buttonPressed);
}

function disableInput() {
    document.getElementById("scramble").disabled = true;
    document.getElementById("solve").disabled = true;
    // 对各个按钮进行解除绑定
    $(document).off("keypress", buttonPressed);
}


function nextState1(moveTimeout=0){
        disableInput();
        //移除数组的第一项并且返回
        enableInput();
		newState = init();
		  //= [27, 30, 15, 34, 4, 32, 9, 12, 2, 29, 25, 0, 19, 13, 23, 24, 5, 35, 6, 50, 33, 46, 22, 16, 47, 10, 51, 38, 48, 42, 41, 31, 39, 45, 52, 44, 11, 28, 20, 43, 40, 7, 53, 3, 36, 26, 21, 17, 1, 49, 37, 8, 14, 18];
		
        //convert back to HTML representation
        //reOrderArray(newState_rep, stateToFE)
		//newState = reOrderArray(newState,stateToFE);
        setStickerColors(newState)
		enableScroll();
		if (moveTimeout != 0) { //check if nextState is used for first_state click, prev_state,etc.
            solveIdx++;
            setSolnText(setColor = true);
        }
}


function nextState(moveTimeout = 0) {
    if (moves.length > 0) {
        disableInput();
        //移除数组的第一项并且返回
        move = moves.shift() // get Move

            //convert to python representation
			// 排序后就是0，1，2，3，4...
            state_rep = reOrderArray(state, FEToState)
            newState_rep = JSON.parse(JSON.stringify(state_rep))

            //swap stickers
            for (var i = 0; i < rotateIdxs_new[move].length; i++) {
                newState_rep[rotateIdxs_new[move][i]] = state_rep[rotateIdxs_old[move][i]]
            }

            // Change move highlight
            if (moveTimeout != 0) { //check if nextState is used for first_state click, prev_state,etc.
                solveIdx++
                setSolnText(setColor = true)
            }

            //convert back to HTML representation
            newState = reOrderArray(newState_rep, stateToFE)
            //set new state
            setStickerColors(newState)

            //Call again if there are more moves
            if (moves.length > 0) {
                setTimeout(function () {
                    nextState(moveTimeout)
                }, moveTimeout);
            } else {
                enableInput();
                if (solveMoves.length > 0) {
                    enableScroll();
                    setSolnText();
                }
            }
    } else {
        enableInput();
        if (solveMoves.length > 0) {
            enableScroll();
            setSolnText();
        }
    }
}


function scrambleCube() {
    disableInput();
    clearSoln();

    numMoves = randInt(100, 200);
    for (var i = 0; i < numMoves; i++) {
        moves.push(legalMoves[randInt(0, legalMoves.length)]);
    }
    nextState(0)
    // nextState1(0);
}

function solveCube() {
    disableInput();
    clearSoln();
    //真正打乱的时候才打开这里
    tempstate = state
    tempstate = reOrderArray(tempstate,FEToState);
    data = {"states":tempstate};
    document.getElementById("solution_text").innerHTML = "SOLVING..."
        $.ajax({
            url: '/solveCube',
            data: {
                "state": JSON.stringify(data)
            },
            type: 'POST',
            dataType: 'json',
            success: function (response) {
                solveStartState = JSON.parse(JSON.stringify(state))

                solveMoves = response["moves"]; //1
                solveMoves_rev = response["moves_rev"]; // 2
                solution_text = response["solve_text"]; //3
                solution_text.push("SOLVED!");
                setSolnText(true);
                moves = JSON.parse(JSON.stringify(solveMoves));

                    setTimeout(function () {
                        nextState(500)
                    }, 500);
            },
            error: function (error) {
                console.log(error);
                document.getElementById("solution_text").innerHTML = "..."
                    setTimeout(function () {
                        solveCube()
                    }, 500);
            },
        });
}
// function solveCube() {
//     disableInput();
//     clearSoln();
//     tempstate = state;
//     tempstate2 = state;
//     tempstate = reOrderArray(tempstate, FEToState);
//     tempstate2 = reOrderArray(tempstate2,stateToFE);
//     data = {"tempstate":tempstate, "tempstate2":tempstate2};
//     //真正打乱的时候才打开这里
//     // state = reOrderArray(state,stateToFE);
//     document.getElementById("solution_text").innerHTML = "SOLVING..."
//         $.ajax({
//             url: '/solveCube',
//             data: {
//                 "state": JSON.stringify(data)
//             },
//             type: 'POST',
//             dataType: 'json',
//             success: function (response) {
//                 solveStartState = JSON.parse(JSON.stringify(state))
//
//                 solveMoves = response["moves"]; //1
//                 solveMoves_rev = response["moves_rev"]; // 2
//                 solution_text = response["solve_text"]; //3
//                 solution_text.push("SOLVED!");
//                 setSolnText(true);
//                 moves = JSON.parse(JSON.stringify(solveMoves));
//
//                     setTimeout(function () {
//                         nextState(500)
//                     }, 500);
//             },
//             error: function (error) {
//                 console.log(error);
//                 document.getElementById("solution_text").innerHTML = "..."
//                     setTimeout(function () {
//                         solveCube()
//                     }, 500);
//             },
//         });
// }

function s1olveCube() {
    disableInput();
    clearSoln();
    //document.getElementById("solution_text").innerHTML = "SOLVING..."
     //   $.ajax({
      //      url: '/solve',
      //      data: {
      //          "state": JSON.stringify(state)
      //      },
     //       type: 'POST',
     ////       dataType: 'json',
       //     success: function (response) {
            //    solveStartState = JSON.parse(JSON.stringify(state))
			//solveStartState = reOrderArray(state,FEToState);	
			
			init("solveCube");
			//魔方官网截取下来的三组数据，能够正常还原
			solveMoves_rev = ['L_-1', 'B_1', 'U_-1', 'D_-1', 'L_1', 'F_1', 'R_-1', 'D_1', 'R_1', 'B_1', 'D_-1', 'B_-1', 'D_1', 'B_-1', 'U_-1', 'F_1', 'L_1', 'F_-1', 'L_-1', 'B_-1', 'L_1', 'U_1', 'F_1', 'U_-1', 'L_-1', 'U_-1'];
			solution_text = ['L', "B'", 'U', 'D', "L'", "F'", 'R', "D'", "R'", "B'", 'D', 'B', "D'", 'B', 'U', "F'", "L'", 'F', 'L', 'B', "L'", "U'", "F'", 'U', 'L', 'U'];
			solveMoves = ['L_1', 'B_-1', 'U_1', 'D_1', 'L_-1', 'F_-1', 'R_1', 'D_-1', 'R_-1', 'B_-1', 'D_1', 'B_1', 'D_-1', 'B_1', 'U_1', 'F_-1', 'L_-1', 'F_1', 'L_1', 'B_1', 'L_-1', 'U_-1', 'F_-1', 'U_1', 'L_1', 'U_1'];

            moves = solveMoves;
			//        solveMoves = response["moves"]; //1
             //   solveMoves_rev = response["moves_rev"]; // 2
              //  solution_text = response["solve_text"]; //3
                solution_text.push("SOLVED!")
                setSolnText(true);

                //moves = JSON.parse(JSON.stringify(solveMoves))

                    //setTimeout(function () {
                        nextState(500);
                   // }, 500);
				  // solveCube();

 }

$(document).ready($(function () {

        disableInput();
        clearSoln();
        $.ajax({
            url: '/stateInit',
            data: {"post":"I want the initData!"},
            type: 'POST',
            dataType: 'json',
            success: function (response) {
                setStickerColors(response["state"]); //  1
                rotateIdxs_old = response["rotateIdxs_old"]; //2
                rotateIdxs_new = response["rotateIdxs_new"]; //3
                stateToFE = response["stateToFE"]; //4
                FEToState = response["FEToState"]; //5
                legalMoves = response["legalMoves"];//6
                enableInput();
            },
            error: function (error) {
                console.log(error);
            },
        });
        $("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)"); //Initial orientation
		//	moves = ["D_1"];
		  //  nextState(0);
		setStickerColors(state);
		enableInput();

        $('#scramble').click(function () {
            scrambleCube()
        });

        $('#solve').click(function () {
            solveCube()
        });

        $('#first_state').click(function () {
            if (solveIdx > 0) {
                moves = solveMoves_rev.slice(0, solveIdx).reverse();
                solveIdx = 0;
                nextState();
            }
        });

        $('#prev_state').click(function () {
            if (solveIdx > 0) {
                solveIdx = solveIdx - 1
                    moves.push(solveMoves_rev[solveIdx])
                    nextState()
            }
        });

        $('#next_state').click(function () {
            if (solveIdx < solveMoves.length) {
                moves.push(solveMoves[solveIdx])
                solveIdx = solveIdx + 1
                    nextState()
            }
        });

        $('#last_state').click(function () {
            if (solveIdx < solveMoves.length) {
                moves = solveMoves.slice(solveIdx, solveMoves.length);
                solveIdx = solveMoves.length
                    nextState();
            }
        });

        $('#cube_div').on("mousedown", function (ev) {
            lastMouseX = ev.clientX;
            lastMouseY = ev.clientY;
            $('#cube_div').on("mousemove", mouseMoved);
        });
        $('#cube_div').on("mouseup", function () {
            $('#cube_div').off("mousemove", mouseMoved);
        });
        $('#cube_div').on("mouseleave", function () {
            $('#cube_div').off("mousemove", mouseMoved);
        });

        console.log("ready!");
    }));

function mouseMoved(ev) {
    var deltaX = ev.pageX - lastMouseX;
    var deltaY = ev.pageY - lastMouseY;

    lastMouseX = ev.pageX;
    lastMouseY = ev.pageY;

    rotY += deltaX * 0.2;
    rotX -= deltaY * 0.5;

    $("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)");
}
