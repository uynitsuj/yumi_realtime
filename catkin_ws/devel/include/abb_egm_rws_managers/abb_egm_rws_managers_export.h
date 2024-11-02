
#ifndef ABB_EGM_RWS_MANAGERS_EXPORT_H
#define ABB_EGM_RWS_MANAGERS_EXPORT_H

#ifdef ABB_EGM_RWS_MANAGERS_STATIC_DEFINE
#  define ABB_EGM_RWS_MANAGERS_EXPORT
#  define ABB_EGM_RWS_MANAGERS_NO_EXPORT
#else
#  ifndef ABB_EGM_RWS_MANAGERS_EXPORT
#    ifdef abb_egm_rws_managers_EXPORTS
        /* We are building this library */
#      define ABB_EGM_RWS_MANAGERS_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define ABB_EGM_RWS_MANAGERS_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef ABB_EGM_RWS_MANAGERS_NO_EXPORT
#    define ABB_EGM_RWS_MANAGERS_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef ABB_EGM_RWS_MANAGERS_DEPRECATED
#  define ABB_EGM_RWS_MANAGERS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef ABB_EGM_RWS_MANAGERS_DEPRECATED_EXPORT
#  define ABB_EGM_RWS_MANAGERS_DEPRECATED_EXPORT ABB_EGM_RWS_MANAGERS_EXPORT ABB_EGM_RWS_MANAGERS_DEPRECATED
#endif

#ifndef ABB_EGM_RWS_MANAGERS_DEPRECATED_NO_EXPORT
#  define ABB_EGM_RWS_MANAGERS_DEPRECATED_NO_EXPORT ABB_EGM_RWS_MANAGERS_NO_EXPORT ABB_EGM_RWS_MANAGERS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ABB_EGM_RWS_MANAGERS_NO_DEPRECATED
#    define ABB_EGM_RWS_MANAGERS_NO_DEPRECATED
#  endif
#endif

#endif /* ABB_EGM_RWS_MANAGERS_EXPORT_H */
