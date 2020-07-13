function predict(engine_id, img_el) {
    var formData = new FormData()
    formData.append('img', $(img_el)[0].files[0])
    formData.append('engine_id', engine_id)
    var result
    $.ajax({
        type: 'post',
        url: '/translation/predict',
        data: formData,
        processData: false,
        contentType: false,
        async: false,
        success: function (res) {
            result = res
        }
    })
    return result
}