Câu 1: Reflex Agent: Hàm evaluation tính dựa vào score của trạng thái kế tiếp - min(koảng cách-> food) + min(koảng cách->ghost)
Câu 2: Gồm 3 hàm: minimax, maxValue(Pacman), minValue(ghost)
    Cả 2 hàm maxValue và minValue đều chajy loop trên các actions có thể từ trạng thái hiện tại, lấy giá trị bằng cách gọi đến hàm minimax. Sau đó voiws hàm maxValue lấy giá trị max của các actions, với hàm minValue lấy giá trị min của actions
    Hàm minimax: Nếu là trạng thái kết thúc->lấy giá trị bằng hàm self.evaluationFunction(). Tiếp tục, kiểm tra xem là pacman hay ghost. Nếu là pacman->maxValue còn ghost->minValue
Câu 3: Tương tự câu 2. Ở hàm maxValue và minValue có thêm phần cắt nhánh:
    maxValue: sau khi cập nhật maxValue,kieermt ra xem maxValue > beta -> cắt nhánh, không thì cập nhật là alpha=maxValue tương tự minValue cũng ậy
Câu 4:Tương tự câu 2 nhưng hàm minValue đổi thành hàm exValue(thay vì tìm min thì value=sum(value của các actions hợp lệ)/số actions hợp lệ
