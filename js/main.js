$(document).ready(function() {
	var i=1; // 변수설정은 함수의 바깥에 설정!
  $("button").click(function() {
    
    $("#option-add").append("<textarea class='form-control' style='font-size: 20px; margin: 10px;' name='prompt' rows='1' id='textbox' placeholder='Type the options here(ex. No)'></textarea>");
    
    i++; // 함수 내 하단에 증가문 설정
    

  });
});