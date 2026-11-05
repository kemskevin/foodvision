output "public_ip" {
  value = aws_instance.vm.public_ip
}

output "ssh" {
  value = "ssh ubuntu@${aws_instance.vm.public_ip}"
}

output "api_url" {
  value = "http://${aws_instance.vm.public_ip}:8080"
}

output "ui_url" {
  value = "http://${aws_instance.vm.public_ip}"
}
