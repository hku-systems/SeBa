#include <ifaddrs.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
int main() {

	  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	    struct sockaddr bind_addr;
	      struct sockaddr_in *addr = (struct sockaddr_in*)&bind_addr;
	        addr->sin_family = AF_INET;
		  addr->sin_addr.s_addr = inet_addr("127.0.0.1");
		    addr->sin_port = 0;

		      if (bind(sockfd, (struct sockaddr *)addr, sizeof(bind_addr)) < 0) {
			              printf("error in bind \n");
				              return -1;
					        }

		        if (listen(sockfd, 3) < 0) {
				        printf("error in listen\n");
					        return -1;
						  }

			  struct sockaddr connect_addr;
			    struct sockaddr_in *addrr = (struct sockaddr_in*)&connect_addr;
			      socklen_t len = sizeof(struct sockaddr);
			        int ret = 0;
				  if ((ret = getsockname(sockfd, &connect_addr, &len)) < 0) {
					          printf("error in getsockname\n");
						          return -1;
							    }

				    printf("Local IP address is: %s:%d\n", inet_ntoa(addrr->sin_addr), ntohs(addrr->sin_port));

				    fork();

				        close(sockfd);
					  printf("success\n");
					    return 0;
}
